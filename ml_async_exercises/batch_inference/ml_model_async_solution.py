#!/usr/bin/env python3
"""
Machine Learning API Batch Inference Exercise - SOLUTION

This script performs batch inference on multiple ML models
hosted as APIs, optimized using async techniques.
"""
import time
import json
import asyncio
import aiohttp
from typing import List, Dict, Any

# Simulated ML model API endpoints (these would be real APIs in practice)
API_ENDPOINTS = {
    "sentiment_model": "https://httpbin.org/delay/1",  # Simulated 1s delay
    "entity_recognition": "https://httpbin.org/delay/0.8",  # Simulated 0.8s delay
    "summarization": "https://httpbin.org/delay/1.5",  # Simulated 1.5s delay
    "classification": "https://httpbin.org/delay/0.7",  # Simulated 0.7s delay
    "question_answering": "https://httpbin.org/delay/1.2",  # Simulated 1.2s delay
}

# Sample input data for inference
def get_batch_data(batch_size: int = 5) -> List[Dict[str, Any]]:
    """Generate sample data for batch inference."""
    return [
        {
            "id": i,
            "text": f"This is sample text {i} for inference.",
            "metadata": {"source": "test", "priority": i % 3}
        }
        for i in range(batch_size)
    ]

# Async implementation for a single API call
async def call_model_api(session: aiohttp.ClientSession, model_name: str, 
                         endpoint: str, text: str) -> Dict[str, Any]:
    """Make an async API call to a single model endpoint."""
    payload = {
        "text": text,
        "model": model_name,
        "parameters": {"max_length": 100}
    }
    
    start_time = time.time()
    
    try:
        async with session.post(endpoint, json=payload) as response:
            duration = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                return {
                    "status": "success",
                    "latency": duration,
                    "data": data
                }
            else:
                return {
                    "status": "error",
                    "latency": duration,
                    "error": f"API Error: {response.status}"
                }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "status": "error",
            "latency": duration,
            "error": f"Exception: {str(e)}"
        }

async def process_input_async(item: Dict[str, Any], 
                              session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Process a single input through all models concurrently."""
    result = {"id": item["id"], "input": item["text"], "model_results": {}}
    
    # Create a task for each model API call
    tasks = []
    for model_name, endpoint in API_ENDPOINTS.items():
        print(f"Queuing {model_name} API for item {item['id']}...")
        task = call_model_api(session, model_name, endpoint, item["text"])
        tasks.append((model_name, task))
    
    # Wait for all API calls to complete concurrently
    for model_name, task in tasks:
        result["model_results"][model_name] = await task
    
    return result

async def batch_inference_async(batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of inputs asynchronously."""
    async with aiohttp.ClientSession() as session:
        # Create a task for each input item
        tasks = [
            process_input_async(item, session)
            for item in batch_data
        ]
        
        # Wait for all items to be processed concurrently
        results = await asyncio.gather(*tasks)
        
    return results

def batch_inference_async_wrapper(batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A wrapper to run the async function in an event loop."""
    return asyncio.run(batch_inference_async(batch_data))

# For reference: Sequential implementation
def process_input_sync(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single input through all models sequentially."""
    import requests
    
    result = {"id": item["id"], "input": item["text"], "model_results": {}}
    
    # Call each model API sequentially
    for model_name, endpoint in API_ENDPOINTS.items():
        print(f"Calling {model_name} API for item {item['id']}...")
        
        # Prepare the payload
        payload = {
            "text": item["text"],
            "model": model_name,
            "parameters": {"max_length": 100}
        }
        
        # Make the API call
        start_time = time.time()
        response = requests.post(endpoint, json=payload)
        duration = time.time() - start_time
        
        # Store the result
        if response.status_code == 200:
            result["model_results"][model_name] = {
                "status": "success",
                "latency": duration,
                "data": response.json()
            }
        else:
            result["model_results"][model_name] = {
                "status": "error",
                "latency": duration,
                "error": f"API Error: {response.status_code}"
            }
    
    return result

def batch_inference_sync(batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of inputs sequentially."""
    results = []
    
    for item in batch_data:
        result = process_input_sync(item)
        results.append(result)
    
    return results

# Main execution
if __name__ == "__main__":
    print("ML Model Batch Inference Solution")
    print("================================")
    
    # Generate test data
    batch_size = 5
    test_data = get_batch_data(batch_size)
    print(f"Generated test batch with {batch_size} items\n")
    
    # Run synchronous version and measure time
    print("Running SYNCHRONOUS batch inference...")
    start_time = time.time()
    sync_results = batch_inference_sync(test_data)
    sync_duration = time.time() - start_time
    print(f"Synchronous execution completed in {sync_duration:.2f} seconds\n")
    
    # Run async implementation
    print("Running ASYNCHRONOUS batch inference...")
    start_time = time.time()
    async_results = batch_inference_async_wrapper(test_data)
    async_duration = time.time() - start_time
    print(f"Asynchronous execution completed in {async_duration:.2f} seconds\n")
    
    # Calculate speedup
    speedup = sync_duration / async_duration
    print(f"Speedup factor: {speedup:.2f}x")
    
    # Print summary of results
    print("\nResults summary:")
    for result in async_results:
        latencies = [v["latency"] for v in result["model_results"].values()]
        avg_latency = sum(latencies) / len(latencies)
        print(f"Item {result['id']}: {len(result['model_results'])} models, "
              f"avg latency {avg_latency:.2f}s")