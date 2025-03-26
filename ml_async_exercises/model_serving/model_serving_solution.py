#!/usr/bin/env python3
"""
ML Model Serving Exercise - SOLUTION

This script demonstrates an async implementation of a model serving system
to improve throughput and reduce latency.
"""
import os
import time
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, Set

# Simulate model storage directory
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Model definitions - in real life these would be actual ML models
MODELS = {
    "text_classifier": {
        "size_mb": 250,
        "load_time": 1.5,  # Seconds to load from disk
        "inference_time": 0.3,  # Seconds per inference
    },
    "image_detector": {
        "size_mb": 550,
        "load_time": 2.8,
        "inference_time": 0.5,
    },
    "recommendation_engine": {
        "size_mb": 800,
        "load_time": 3.2,
        "inference_time": 0.7,
    },
    "sentiment_analyzer": {
        "size_mb": 120,
        "load_time": 0.8,
        "inference_time": 0.2,
    },
    "entity_extractor": {
        "size_mb": 350,
        "load_time": 1.7,
        "inference_time": 0.4,
    }
}

class Request:
    """Simulate an inference request."""
    def __init__(self, request_id: str, model_name: str, inputs: List[Any]):
        self.request_id = request_id
        self.model_name = model_name
        self.inputs = inputs
        self.timestamp = time.time()
    
    def __repr__(self):
        return f"Request(id={self.request_id}, model={self.model_name}, inputs={len(self.inputs)})"

# Original synchronous implementation for comparison
class ModelServer:
    """Synchronous model server implementation."""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.cache = {}
        self.max_cache_size_mb = 1000  # 1GB cache limit
        self.current_cache_size_mb = 0
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """
        Load a model into memory from disk or cache.
        In a real implementation, this would load actual model weights.
        """
        # Check if model is already in cache
        if model_name in self.cache:
            print(f"Model {model_name} found in cache")
            return self.cache[model_name]
        
        # Check if model exists
        if model_name not in MODELS:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = MODELS[model_name]
        
        # Simulate loading model from disk
        print(f"Loading model {model_name} from disk...")
        time.sleep(model_info["load_time"])  # Simulate loading time
        
        # Create model object
        model = {
            "name": model_name,
            "size_mb": model_info["size_mb"],
            "loaded_at": time.time(),
            "metadata": {
                "inference_time": model_info["inference_time"]
            }
        }
        
        # Add to cache if there's room
        if self.current_cache_size_mb + model_info["size_mb"] <= self.max_cache_size_mb:
            self.cache[model_name] = model
            self.current_cache_size_mb += model_info["size_mb"]
            print(f"Added {model_name} to cache. Cache size: {self.current_cache_size_mb}MB")
        else:
            print(f"Cache full, {model_name} will not be cached")
        
        return model
    
    def run_inference(self, model: Dict[str, Any], inputs: List[Any]) -> List[Any]:
        """
        Run inference on the given inputs using the loaded model.
        In a real implementation, this would use the model to generate predictions.
        """
        # Simulate inference computation
        inference_time = model["metadata"]["inference_time"]
        
        # Scale inference time by batch size
        batch_size = len(inputs)
        total_inference_time = inference_time * (1 + 0.1 * (batch_size - 1))
        
        print(f"Running inference with {model['name']} on {batch_size} inputs...")
        time.sleep(total_inference_time)  # Simulate computation time
        
        # Generate dummy predictions
        predictions = [
            {"input_id": i, "score": random.random(), "latency": total_inference_time}
            for i in range(batch_size)
        ]
        
        return predictions
    
    def process_request(self, request: Request) -> Dict[str, Any]:
        """
        Process a single inference request:
        1. Load the requested model
        2. Run inference
        3. Return results
        """
        start_time = time.time()
        
        try:
            # Load model
            model = self.load_model(request.model_name)
            
            # Run inference
            predictions = self.run_inference(model, request.inputs)
            
            # Prepare response
            response = {
                "request_id": request.request_id,
                "model_name": request.model_name,
                "predictions": predictions,
                "input_count": len(request.inputs),
                "latency": time.time() - start_time
            }
            
            return response
        
        except Exception as e:
            return {
                "request_id": request.request_id,
                "error": str(e),
                "latency": time.time() - start_time
            }
    
    def process_batch(self, requests: List[Request]) -> List[Dict[str, Any]]:
        """Process a batch of requests sequentially."""
        results = []
        for request in requests:
            result = self.process_request(request)
            results.append(result)
        return results

# ASYNC IMPLEMENTATION
class AsyncModelServer:
    """Asynchronous model server implementation."""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.cache = {}
        self.max_cache_size_mb = 1000  # 1GB cache limit
        self.current_cache_size_mb = 0
        self.model_load_locks = {}  # Prevent duplicate loads
    
    async def load_model_async(self, model_name: str) -> Dict[str, Any]:
        """
        Load a model asynchronously from disk or cache.
        Uses locks to prevent the same model from being loaded multiple times.
        """
        # Check if model is already in cache
        if model_name in self.cache:
            print(f"Model {model_name} found in cache")
            return self.cache[model_name]
        
        # Check if model exists
        if model_name not in MODELS:
            raise ValueError(f"Model {model_name} not found")
        
        # Use a lock to prevent duplicate loading of the same model
        if model_name not in self.model_load_locks:
            self.model_load_locks[model_name] = asyncio.Lock()
        
        # Acquire lock for this model
        async with self.model_load_locks[model_name]:
            # Check cache again in case another task loaded it while we were waiting
            if model_name in self.cache:
                print(f"Model {model_name} loaded by another task")
                return self.cache[model_name]
            
            model_info = MODELS[model_name]
            
            # Simulate loading model from disk
            print(f"Loading model {model_name} from disk...")
            await asyncio.sleep(model_info["load_time"])  # Non-blocking sleep
            
            # Create model object
            model = {
                "name": model_name,
                "size_mb": model_info["size_mb"],
                "loaded_at": time.time(),
                "metadata": {
                    "inference_time": model_info["inference_time"]
                }
            }
            
            # Add to cache if there's room
            if self.current_cache_size_mb + model_info["size_mb"] <= self.max_cache_size_mb:
                self.cache[model_name] = model
                self.current_cache_size_mb += model_info["size_mb"]
                print(f"Added {model_name} to cache. Cache size: {self.current_cache_size_mb}MB")
            else:
                print(f"Cache full, {model_name} will not be cached")
            
            return model
    
    async def run_inference_async(self, model: Dict[str, Any], inputs: List[Any]) -> List[Any]:
        """
        Run inference asynchronously on the given inputs using the loaded model.
        In a real-world scenario, we could also run multiple inferences in parallel.
        """
        # Simulate inference computation
        inference_time = model["metadata"]["inference_time"]
        
        # Scale inference time by batch size
        batch_size = len(inputs)
        total_inference_time = inference_time * (1 + 0.1 * (batch_size - 1))
        
        print(f"Running inference with {model['name']} on {batch_size} inputs...")
        
        # For CPU-bound inference in real systems, we would use run_in_executor
        # Here we use asyncio.sleep to simulate without blocking
        await asyncio.sleep(total_inference_time)
        
        # Generate dummy predictions
        predictions = [
            {"input_id": i, "score": random.random(), "latency": total_inference_time}
            for i in range(batch_size)
        ]
        
        return predictions
    
    async def process_request_async(self, request: Request) -> Dict[str, Any]:
        """Process a single inference request asynchronously."""
        start_time = time.time()
        
        try:
            # Load model
            model = await self.load_model_async(request.model_name)
            
            # Run inference
            predictions = await self.run_inference_async(model, request.inputs)
            
            # Prepare response
            response = {
                "request_id": request.request_id,
                "model_name": request.model_name,
                "predictions": predictions,
                "input_count": len(request.inputs),
                "latency": time.time() - start_time
            }
            
            return response
        
        except Exception as e:
            return {
                "request_id": request.request_id,
                "error": str(e),
                "latency": time.time() - start_time
            }
    
    async def process_batch_async(self, requests: List[Request]) -> List[Dict[str, Any]]:
        """Process a batch of requests concurrently."""
        tasks = [self.process_request_async(request) for request in requests]
        return await asyncio.gather(*tasks)

def process_batch_async_wrapper(requests: List[Request]) -> List[Dict[str, Any]]:
    """Wrapper to run async batch processing."""
    return asyncio.run(AsyncModelServer(MODEL_DIR).process_batch_async(requests))

def generate_test_requests(num_requests: int) -> List[Request]:
    """Generate a batch of test requests."""
    requests = []
    for i in range(num_requests):
        model_name = random.choice(list(MODELS.keys()))
        batch_size = random.randint(1, 5)  # Random batch size
        inputs = [f"input_{i}_{j}" for j in range(batch_size)]
        request = Request(f"req_{i}", model_name, inputs)
        requests.append(request)
    return requests

if __name__ == "__main__":
    print("ML Model Serving Solution")
    print("========================")
    
    # Generate test requests
    num_requests = 10
    test_requests = generate_test_requests(num_requests)
    print(f"Generated {num_requests} test requests")
    
    # Process requests synchronously
    print("\nProcessing requests with SYNCHRONOUS server...")
    server = ModelServer(MODEL_DIR)
    
    start_time = time.time()
    sync_results = server.process_batch(test_requests)
    sync_duration = time.time() - start_time
    
    print(f"Synchronous processing completed in {sync_duration:.2f} seconds")
    
    # Calculate average latency
    avg_latency = sum(r["latency"] for r in sync_results) / len(sync_results)
    print(f"Average request latency: {avg_latency:.2f} seconds")
    
    # Process requests asynchronously
    print("\nProcessing requests with ASYNCHRONOUS server...")
    start_time = time.time()
    async_results = process_batch_async_wrapper(test_requests)
    async_duration = time.time() - start_time
    
    print(f"Asynchronous processing completed in {async_duration:.2f} seconds")
    
    # Calculate average latency
    avg_latency_async = sum(r["latency"] for r in async_results) / len(async_results)
    print(f"Average request latency: {avg_latency_async:.2f} seconds")
    
    # Calculate speedup
    speedup = sync_duration / async_duration
    print(f"Speedup factor: {speedup:.2f}x")