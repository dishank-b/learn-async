#!/usr/bin/env python3
"""
ML Model Serving Exercise

This script simulates a simple model serving API that:
1. Handles incoming requests
2. Loads models from a cache or disk
3. Runs inference
4. Returns predictions

Your task: Convert the synchronous implementation to async
to improve throughput and latency.
"""
import os
import time
import json
import random
from typing import Dict, List, Any, Optional

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

# In-memory model cache
model_cache = {}

class Request:
    """Simulate an inference request."""
    def __init__(self, request_id: str, model_name: str, inputs: List[Any]):
        self.request_id = request_id
        self.model_name = model_name
        self.inputs = inputs
        self.timestamp = time.time()
    
    def __repr__(self):
        return f"Request(id={self.request_id}, model={self.model_name}, inputs={len(self.inputs)})"

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

# TODO: IMPLEMENT ASYNC VERSION HERE
# You'll need to implement:
# 1. class AsyncModelServer that handles async loading and inference
# 2. async def load_model_async(self, model_name)
# 3. async def run_inference_async(self, model, inputs)
# 4. async def process_request_async(self, request)
# 5. async def process_batch_async(self, requests)

# def process_batch_async_wrapper(requests):
#     """Wrapper to run async batch processing."""
#     import asyncio
#     return asyncio.run(AsyncModelServer(MODEL_DIR).process_batch_async(requests))

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
    print("ML Model Serving Exercise")
    print("=========================")
    
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
    
    # TODO: Uncomment to run your async implementation
    # print("\nProcessing requests with ASYNCHRONOUS server...")
    # start_time = time.time()
    # async_results = process_batch_async_wrapper(test_requests)
    # async_duration = time.time() - start_time
    # 
    # print(f"Asynchronous processing completed in {async_duration:.2f} seconds")
    # 
    # # Calculate average latency
    # avg_latency_async = sum(r["latency"] for r in async_results) / len(async_results)
    # print(f"Average request latency: {avg_latency_async:.2f} seconds")
    # 
    # # Calculate speedup
    # speedup = sync_duration / async_duration
    # print(f"Speedup factor: {speedup:.2f}x")