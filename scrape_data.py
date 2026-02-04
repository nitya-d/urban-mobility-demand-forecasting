"""
TfL Bike Journey Data Scraper

Downloads bike journey data from Transport for London's cycling statistics page.
Saves data as parquet files for efficient storage and loading.

Usage:
    python scrape_data.py              # Download all years (2019-2021)
    python scrape_data.py --year 2019  # Download specific year
"""

import argparse
import re
import requests
from io import StringIO
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def scrape_tfl_usage_csvs(
    url: str = "https://cycling.data.tfl.gov.uk/#!usage-stats%2F"
) -> list[str]:
    """
    Scrape TfL cycling usage-stats page using Selenium (handles JavaScript).
    Returns list of JourneyDataExtract CSV URLs.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        # Wait for links to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, "JourneyDataExtract"))
        )
        
        links = driver.find_elements(By.TAG_NAME, "a")
        urls = []
        for link in links:
            href = link.get_attribute("href")
            if href and "JourneyDataExtract" in href and href.endswith(".csv"):
                urls.append(href)
        
        if not urls:
            raise RuntimeError("No JourneyDataExtract CSVs found")
        
        return sorted(set(urls))
    finally:
        driver.quit()


def filter_urls_by_year(csv_urls: list[str], year: int) -> list[str]:
    """
    Filter CSV URLs to only include files where the START date matches the given year.
    E.g., Dec2018-Jan2019 belongs to 2018, not 2019.
    
    Args:
        csv_urls: List of CSV URLs from scrape_tfl_usage_csvs()
        year: The year to filter by (e.g., 2019)
        
    Returns:
        List of URLs matching the specified year
    """
    pattern = re.compile(r"JourneyDataExtract(\d{2}[A-Za-z]{3}\d{4})-")
    filtered = []
    
    for url in csv_urls:
        filename = url.split("/")[-1]
        match = pattern.search(filename)
        
        if not match:
            continue
            
        start_date_str = match.group(1)
        try:
            start_date = datetime.strptime(start_date_str, "%d%b%Y")
            if start_date.year == year:
                filtered.append(url)
        except ValueError:
            continue
    
    return filtered


def download_csv(url: str, headers: dict) -> tuple[str, pd.DataFrame | None]:
    """Download a single CSV and return (filename, dataframe)."""
    filename = url.split("/")[-1]
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        return filename, df
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return filename, None


def download_year_to_parquet_parallel(
    year: int, 
    csv_urls: list[str], 
    output_dir: str = "data",
    max_workers: int = 10
) -> Path:
    """
    Download all CSVs for a given year in parallel and save as a single parquet file.
    
    Args:
        year: The year to download (e.g., 2019)
        csv_urls: List of CSV URLs from scrape_tfl_usage_csvs()
        output_dir: Directory to save the parquet file
        max_workers: Number of parallel downloads (default 10)
        
    Returns:
        Path to the saved parquet file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Filter URLs first
    year_urls = filter_urls_by_year(csv_urls, year)
    
    if not year_urls:
        raise ValueError(f"No files found for year {year}")
    
    print(f"Found {len(year_urls)} files for {year}. Downloading with {max_workers} threads...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    dfs = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_csv, url, headers): url for url in year_urls}
        
        for future in as_completed(futures):
            filename, df = future.result()
            if df is not None:
                dfs.append(df)
                print(f"✓ {filename}")
            else:
                failed.append(filename)
                print(f"✗ {filename}")
    
    if failed:
        print(f"\nFailed to download {len(failed)} files: {failed}")
    
    if not dfs:
        raise ValueError(f"Failed to download any files for year {year}")
    
    result = pd.concat(dfs, ignore_index=True)
    
    # Convert date columns if they exist
    date_cols = ["Start Date", "End Date"]
    for col in date_cols:
        if col in result.columns:
            result[col] = pd.to_datetime(result[col], dayfirst=True)
    
    parquet_path = output_path / f"journeys_{year}.parquet"
    result.to_parquet(parquet_path, index=False)
    print(f"\n✓ Saved {len(result):,} rows from {len(dfs)} files to {parquet_path}")
    
    return parquet_path


def combine_years(years: list[int], output_dir: str = "data") -> Path:
    """
    Combine individual year parquet files into a single optimized file.
    
    Args:
        years: List of years to combine
        output_dir: Directory containing parquet files
        
    Returns:
        Path to combined parquet file
    """
    output_path = Path(output_dir)
    dfs = []
    
    for year in years:
        parquet_path = output_path / f"journeys_{year}.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            dfs.append(df)
            print(f"✓ Loaded {year}: {len(df):,} rows")
        else:
            print(f"✗ Missing {parquet_path}")
    
    if not dfs:
        raise ValueError("No parquet files found to combine")
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Remove records outside date range
    df = df[df['Start Date'].dt.year.isin(years)]
    
    # Optimize memory usage
    if 'Rental Id' in df.columns:
        df = df.drop(columns=['Rental Id'])
    
    # Convert duration to minutes
    df['Duration'] = (df['Duration'] / 60).astype('float32')
    
    # Add time features
    df['DayOfWeek'] = df['Start Date'].dt.day_name().astype('category')
    df['Date'] = df['Start Date'].dt.date.astype('datetime64[ns]')
    df['Year'] = df['Start Date'].dt.year.astype('category')
    df['Start Hour'] = df['Start Date'].dt.hour.astype('uint8')
    
    # Convert strings to category for memory efficiency
    for col in ['EndStation Name', 'StartStation Name', 'Bike Id', 
                'EndStation Id', 'StartStation Id']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    combined_path = output_path / f"journeys_{'_'.join(map(str, years))}.parquet"
    df.to_parquet(combined_path, engine="pyarrow", index=False)
    print(f"\n✓ Combined {len(df):,} rows saved to {combined_path}")
    
    return combined_path


def main():
    parser = argparse.ArgumentParser(
        description="Download TfL bike journey data"
    )
    parser.add_argument(
        "--year", 
        type=int, 
        help="Specific year to download (default: all 2019-2021)"
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Only combine existing parquet files, don't download"
    )
    args = parser.parse_args()
    
    years = [2019, 2020, 2021]
    
    if args.combine_only:
        combine_years(years)
        return
    
    print("Scraping TfL website for CSV URLs...")
    csv_urls = scrape_tfl_usage_csvs()
    print(f"Found {len(csv_urls)} total CSV files\n")
    
    if args.year:
        download_year_to_parquet_parallel(args.year, csv_urls)
    else:
        for year in years:
            print(f"\n{'='*50}")
            print(f"Downloading {year}")
            print('='*50)
            download_year_to_parquet_parallel(year, csv_urls)
        
        print(f"\n{'='*50}")
        print("Combining all years...")
        print('='*50)
        combine_years(years)


if __name__ == "__main__":
    main()
