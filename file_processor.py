#!/usr/bin/env python3
"""
Async file processing example.
Shows how to process files asynchronously using asyncio with real file I/O.
"""
import asyncio
import os
import time
from pathlib import Path

# Function to create test files for our demonstration
def create_test_files(directory: str, num_files: int, size_kb: int):
    """Create test files for the demonstration."""
    os.makedirs(directory, exist_ok=True)
    
    content = b"x" * 1024  # 1KB of data
    
    for i in range(num_files):
        filename = os.path.join(directory, f"testfile_{i}.txt")
        with open(filename, "wb") as f:
            for _ in range(size_kb):
                f.write(content)  # Write in 1KB chunks
    
    print(f"Created {num_files} test files of {size_kb}KB each in {directory}")

# Synchronous file processing function
def process_file_sync(filename: str):
    """Process a file synchronously."""
    print(f"Starting to process {filename}...")
    
    # Read the file
    with open(filename, "rb") as f:
        content = f.read()
    
    # Simulate some CPU-intensive processing
    time.sleep(0.2)  # Simulate processing time
    file_size = len(content)
    result = f"File {os.path.basename(filename)}: {file_size} bytes"
    
    print(f"Finished processing {filename}")
    return result

# Process all files synchronously
def process_files_sync(directory: str):
    """Process all files in a directory synchronously."""
    print("\nProcessing files synchronously...")
    start = time.time()
    
    results = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            result = process_file_sync(filepath)
            results.append(result)
    
    end = time.time()
    print(f"Synchronous processing completed in {end - start:.2f} seconds")
    return results

# Async file processing function
async def process_file_async(filename: str, loop):
    """Process a file asynchronously."""
    print(f"Starting to process {filename} asynchronously...")
    
    # Read the file using an executor (for file I/O)
    # This moves the blocking file I/O to a separate thread
    content = await loop.run_in_executor(None, lambda: open(filename, "rb").read())
    
    # Simulate some CPU-intensive processing in another thread
    await asyncio.sleep(0.2)  # Simulate processing time
    file_size = len(content)
    result = f"File {os.path.basename(filename)}: {file_size} bytes"
    
    print(f"Finished processing {filename} asynchronously")
    return result

# Process all files asynchronously
async def process_files_async(directory: str):
    """Process all files in a directory asynchronously."""
    print("\nProcessing files asynchronously...")
    start = time.time()
    
    # Get the event loop
    loop = asyncio.get_running_loop()
    
    # Create tasks for all files
    tasks = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            task = process_file_async(filepath, loop)
            tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    end = time.time()
    print(f"Asynchronous processing completed in {end - start:.2f} seconds")
    return results

# Main function
async def main():
    print("FILE PROCESSING DEMONSTRATION\n")
    
    # Create test data directory and files
    test_dir = "test_files"
    create_test_files(test_dir, 10, 100)  # 10 files, 100KB each
    
    try:
        # Process files synchronously
        sync_results = process_files_sync(test_dir)
        for result in sync_results:
            print(result)
        
        # Process files asynchronously
        async_results = await process_files_async(test_dir)
        print("\nAsynchronous results:")
        for result in async_results:
            print(result)
        
        print("\nNote: The asynchronous version is faster because file I/O operations")
        print("are moved to a thread pool, allowing multiple files to be processed")
        print("concurrently without blocking the main event loop.")
    finally:
        # Clean up (optional)
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\nCleaned up test directory: {test_dir}")

if __name__ == "__main__":
    asyncio.run(main())