#!/usr/bin/env python3
"""
An async web scraper example.
This demonstrates using asyncio with HTTP requests - a common real-world use case.
"""
import asyncio
import time
import aiohttp  # You may need to install this: pip install aiohttp

# List of URLs to fetch
URLS = [
    "https://example.com",
    "https://httpbin.org/delay/1",  # This URL has a 1-second delay
    "https://httpbin.org/delay/2",  # This URL has a 2-second delay
    "https://python.org",
    "https://httpbin.org/delay/1",  # Another with 1-second delay
]

# Sequential version for comparison
def fetch_urls_sync():
    """Fetch URLs sequentially using requests."""
    import requests  # We import here to avoid unused import if only async is used
    
    print("Fetching URLs synchronously...")
    start = time.time()
    results = []
    
    for url in URLS:
        print(f"Fetching: {url}")
        response = requests.get(url)
        results.append((url, len(response.text)))
    
    end = time.time()
    print(f"Synchronous fetch completed in {end - start:.2f} seconds")
    return results

# Async version
async def fetch_url(session, url):
    """Fetch a single URL asynchronously."""
    print(f"Starting fetch: {url}")
    async with session.get(url) as response:
        content = await response.text()
        print(f"Finished fetch: {url}")
        return url, len(content)

async def fetch_urls_async():
    """Fetch all URLs concurrently."""
    print("\nFetching URLs asynchronously...")
    start = time.time()
    
    # Create a shared session
    async with aiohttp.ClientSession() as session:
        # Create tasks for all URLs
        tasks = [fetch_url(session, url) for url in URLS]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
    
    end = time.time()
    print(f"Asynchronous fetch completed in {end - start:.2f} seconds")
    return results

# Main function
async def main():
    print("WEB SCRAPER EXAMPLE\n")
    print("This example demonstrates the difference between synchronous and asynchronous HTTP requests.")
    print("Note: This requires 'requests' and 'aiohttp' packages installed.")
    
    # Run synchronous version
    try:
        sync_results = fetch_urls_sync()
        for url, size in sync_results:
            print(f"URL: {url} - Content size: {size} bytes")
    except ImportError:
        print("Could not run synchronous example: 'requests' package not installed")
    
    # Run asynchronous version
    try:
        async_results = await fetch_urls_async()
        print("\nAsynchronous results:")
        for url, size in async_results:
            print(f"URL: {url} - Content size: {size} bytes")
        
        print("\nNote: The asynchronous version is much faster because it processes multiple")
        print("requests concurrently, without waiting for each response before starting the next.")
    except ImportError:
        print("Could not run asynchronous example: 'aiohttp' package not installed")

if __name__ == "__main__":
    asyncio.run(main())  # Python 3.7+ way to run async code