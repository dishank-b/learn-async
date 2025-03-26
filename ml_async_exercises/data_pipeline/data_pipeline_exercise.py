#!/usr/bin/env python3
"""
ML Data Pipeline Exercise

This script simulates ETL operations for an ML training pipeline:
1. Downloads data files from multiple sources
2. Processes each file
3. Combines results into a single dataset
4. Caches the results

Your task: Convert the synchronous implementation to async
to improve performance.
"""
import os
import time
import json
import shutil
import requests
from typing import List, Dict, Any

# Simulate remote data sources (in real life these would be real URLs)
DATA_SOURCES = {
    "user_data": "https://httpbin.org/delay/1.5",
    "product_data": "https://httpbin.org/delay/0.8",
    "interaction_data": "https://httpbin.org/delay/2.1",
    "context_data": "https://httpbin.org/delay/1.2",
    "feature_data": "https://httpbin.org/delay/0.9",
}

# Cache directory for downloaded data
CACHE_DIR = "./data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def download_dataset(dataset_name: str, url: str) -> Dict[str, Any]:
    """
    Download a dataset from a remote source.
    Returns the downloaded data as a JSON object.
    """
    print(f"Downloading {dataset_name} from {url}...")
    
    # Check cache first
    cache_file = os.path.join(CACHE_DIR, f"{dataset_name}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            print(f"  Loading {dataset_name} from cache")
            return json.load(f)
    
    # Download the data
    start_time = time.time()
    response = requests.get(url, params={"dataset": dataset_name})
    duration = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        # Add metadata
        result = {
            "name": dataset_name,
            "download_time": duration,
            "timestamp": time.time(),
            "data": data
        }
        
        # Cache the result
        with open(cache_file, 'w') as f:
            json.dump(result, f)
            
        print(f"  Downloaded {dataset_name} in {duration:.2f}s")
        return result
    else:
        print(f"  Error downloading {dataset_name}: {response.status_code}")
        return {"name": dataset_name, "error": f"HTTP {response.status_code}"}

def process_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single dataset with simulated compute operations.
    """
    dataset_name = dataset["name"]
    print(f"Processing {dataset_name}...")
    
    # Simulate processing time based on dataset size
    process_time = 0.5  # Base processing time
    if "data" in dataset:
        # Simulate more processing for larger datasets
        process_time += len(json.dumps(dataset["data"])) / 50000
    
    # Simulate compute-intensive operation
    start_time = time.time()
    time.sleep(process_time)  # In a real scenario, this would be actual computation
    duration = time.time() - start_time
    
    result = {
        "name": dataset_name,
        "processed": True,
        "process_time": duration,
        "timestamp": time.time(),
        "data": dataset.get("data", {})
    }
    
    print(f"  Processed {dataset_name} in {duration:.2f}s")
    return result

def combine_datasets(processed_datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine multiple processed datasets into a single training dataset.
    """
    print("Combining datasets...")
    start_time = time.time()
    
    # Simulate combination logic
    combined = {
        "datasets": [ds["name"] for ds in processed_datasets],
        "total_datasets": len(processed_datasets),
        "features": {},
        "timestamp": time.time()
    }
    
    # Merge data from all datasets (simplified)
    for ds in processed_datasets:
        if "data" in ds:
            for key, value in ds.get("data", {}).items():
                if key not in combined["features"]:
                    combined["features"][key] = []
                combined["features"][key].append(value)
    
    # Simulate some combination work
    time.sleep(0.5)
    duration = time.time() - start_time
    
    print(f"Combined {len(processed_datasets)} datasets in {duration:.2f}s")
    return combined

def etl_pipeline_sync() -> Dict[str, Any]:
    """
    Run the complete ETL pipeline synchronously.
    """
    # Step 1: Download all datasets
    downloaded_datasets = []
    for name, url in DATA_SOURCES.items():
        dataset = download_dataset(name, url)
        downloaded_datasets.append(dataset)
    
    # Step 2: Process each dataset
    processed_datasets = []
    for dataset in downloaded_datasets:
        processed = process_dataset(dataset)
        processed_datasets.append(processed)
    
    # Step 3: Combine into final dataset
    final_dataset = combine_datasets(processed_datasets)
    
    return final_dataset

# TODO: IMPLEMENT ASYNC VERSION
# You'll need to create:
# 1. async def download_dataset_async(name, url, session)
# 2. async def process_dataset_async(dataset)
# 3. async def etl_pipeline_async()

# def run_async_pipeline():
#     """Wrapper to run the async pipeline"""
#     import asyncio
#     return asyncio.run(etl_pipeline_async())

def clear_cache():
    """Clear the data cache directory"""
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)

if __name__ == "__main__":
    print("ML Data Pipeline Exercise")
    print("========================")
    
    # Clear cache to ensure we download fresh data
    clear_cache()
    
    # Run synchronous pipeline and measure time
    print("\nRunning SYNCHRONOUS pipeline...")
    start_time = time.time()
    result_sync = etl_pipeline_sync()
    sync_duration = time.time() - start_time
    print(f"Synchronous pipeline completed in {sync_duration:.2f} seconds")
    
    # Clear cache again for fair comparison
    clear_cache()
    
    # TODO: Uncomment to test your async implementation
    # print("\nRunning ASYNCHRONOUS pipeline...")
    # start_time = time.time()
    # result_async = run_async_pipeline()
    # async_duration = time.time() - start_time
    # print(f"Asynchronous pipeline completed in {async_duration:.2f} seconds")
    # 
    # # Calculate speedup
    # speedup = sync_duration / async_duration
    # print(f"Speedup factor: {speedup:.2f}x")
    
    # Simple validation of results
    # Uncomment this once you've implemented the async version
    # print("\nValidation:")
    # print(f"Sync pipeline processed {result_sync['total_datasets']} datasets")
    # print(f"Async pipeline processed {result_async['total_datasets']} datasets")
    # print(f"Results match: {result_sync['total_datasets'] == result_async['total_datasets']}")
    
    # Clean up
    clear_cache()