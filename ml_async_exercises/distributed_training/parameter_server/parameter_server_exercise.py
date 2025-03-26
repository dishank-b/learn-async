#!/usr/bin/env python3
"""
Distributed Training with Parameter Server Exercise

This script simulates a distributed training system with a parameter server architecture:
1. A central parameter server maintains the global model
2. Multiple worker nodes compute gradients on local data
3. Workers push gradients to the parameter server
4. Workers pull updated parameters from the server

Your task: Convert the synchronous implementation to async for better performance
and more realistic coordination between nodes.
"""
import time
import random
import json
import numpy as np
from typing import Dict, List, Any, Tuple

# Constants
NUM_WORKERS = 4
NUM_ITERATIONS = 20
LAYER_SIZES = [784, 128, 10]  # Input -> Hidden -> Output
LEARNING_RATE = 0.01
SIMULATE_NETWORK_LATENCY = True

class NeuralNetworkModel:
    """Simulated neural network model with weights and biases."""
    
    def __init__(self, layer_sizes):
        """Initialize model with random weights."""
        self.weights = []
        self.biases = []
        self.layer_sizes = layer_sizes
        
        # Initialize random weights and biases
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.biases.append(np.random.randn(layer_sizes[i+1]) * 0.1)
    
    def get_params(self) -> Dict[str, List[np.ndarray]]:
        """Get current model parameters."""
        return {
            "weights": self.weights.copy(),
            "biases": self.biases.copy()
        }
    
    def set_params(self, params: Dict[str, List[np.ndarray]]) -> None:
        """Set model parameters."""
        self.weights = params["weights"].copy()
        self.biases = params["biases"].copy()
    
    def apply_gradients(self, gradients: Dict[str, List[np.ndarray]]) -> None:
        """Apply gradients to update model parameters."""
        weight_gradients = gradients["weights"]
        bias_gradients = gradients["biases"]
        
        for i in range(len(self.weights)):
            self.weights[i] -= LEARNING_RATE * weight_gradients[i]
            self.biases[i] -= LEARNING_RATE * bias_gradients[i]

class ParameterServer:
    """Central server that maintains the global model parameters."""
    
    def __init__(self, layer_sizes: List[int]):
        """Initialize parameter server with a model."""
        self.model = NeuralNetworkModel(layer_sizes)
        self.update_count = 0
        self.total_worker_time = 0  # Track cumulative compute time
    
    def get_parameters(self) -> Dict[str, List[np.ndarray]]:
        """Return current model parameters."""
        if SIMULATE_NETWORK_LATENCY:
            # Simulate network latency for parameter transfer
            time.sleep(0.05)
        return self.model.get_params()
    
    def apply_gradients(self, gradients: Dict[str, List[np.ndarray]], worker_time: float) -> None:
        """
        Apply gradients from a worker to update the model.
        Also track worker compute time for metrics.
        """
        if SIMULATE_NETWORK_LATENCY:
            # Simulate network latency for gradient transfer
            time.sleep(0.05)
            
        # Apply the gradients to update the model
        self.model.apply_gradients(gradients)
        self.update_count += 1
        self.total_worker_time += worker_time
        
        print(f"Parameter Server: Applied update {self.update_count}. "
              f"Total worker compute time: {self.total_worker_time:.2f}s")

class Worker:
    """Worker node that computes gradients on local data."""
    
    def __init__(self, worker_id: int, layer_sizes: List[int]):
        """Initialize worker with an ID and local model."""
        self.worker_id = worker_id
        self.local_model = NeuralNetworkModel(layer_sizes)
        self.iteration = 0
        self.compute_times = []
    
    def generate_fake_data_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a fake data batch for training."""
        # For simplicity, we'll just generate random data
        batch_size = random.randint(16, 64)
        x = np.random.randn(batch_size, LAYER_SIZES[0])
        y = np.random.randint(0, LAYER_SIZES[-1], batch_size)
        return x, y
    
    def compute_gradients(self, params: Dict[str, List[np.ndarray]]) -> Tuple[Dict[str, List[np.ndarray]], float]:
        """
        Compute gradients using local data.
        Returns gradients and computation time.
        """
        # Set local model parameters to the ones received from the server
        self.local_model.set_params(params)
        
        # Generate a fake data batch
        x, y = self.generate_fake_data_batch()
        
        # Simulate computation time - workers are heterogeneous
        # Some are faster, some are slower
        compute_time = 0.1 + 0.2 * np.abs(np.sin(self.worker_id * 0.5)) + 0.05 * random.random()
        time.sleep(compute_time)  # Simulate computation
        
        # Generate fake gradients (in a real system, these would be computed via backprop)
        weight_gradients = []
        bias_gradients = []
        
        for w, b in zip(self.local_model.weights, self.local_model.biases):
            # Generate random gradients scaled by a decaying factor to simulate convergence
            decay_factor = 1.0 / (1.0 + 0.1 * self.iteration)
            weight_gradients.append(np.random.randn(*w.shape) * 0.1 * decay_factor)
            bias_gradients.append(np.random.randn(*b.shape) * 0.1 * decay_factor)
        
        gradients = {
            "weights": weight_gradients,
            "biases": bias_gradients
        }
        
        self.iteration += 1
        self.compute_times.append(compute_time)
        
        print(f"Worker {self.worker_id}: Computed gradients (iteration {self.iteration}, "
              f"batch size: {len(x)}, compute time: {compute_time:.2f}s)")
        
        return gradients, compute_time

def train_synchronous(parameter_server: ParameterServer, workers: List[Worker], num_iterations: int) -> Dict[str, Any]:
    """
    Perform synchronous distributed training.
    
    In synchronous SGD:
    1. Workers fetch parameters from the server
    2. Each worker computes gradients on its local data
    3. Server waits for ALL workers to finish before applying updates
    4. Repeat
    """
    print("\nStarting SYNCHRONOUS distributed training...")
    
    start_time = time.time()
    metrics = {"iterations": [], "times": []}
    
    for iteration in range(num_iterations):
        iteration_start = time.time()
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Get current parameters from the server
        params = parameter_server.get_parameters()
        
        # Each worker computes gradients
        all_gradients = []
        total_compute_time = 0
        
        for worker in workers:
            gradients, compute_time = worker.compute_gradients(params)
            all_gradients.append(gradients)
            total_compute_time += compute_time
        
        # Average the gradients from all workers
        avg_weight_gradients = []
        avg_bias_gradients = []
        
        for i in range(len(all_gradients[0]["weights"])):
            avg_weight = np.mean([g["weights"][i] for g in all_gradients], axis=0)
            avg_bias = np.mean([g["biases"][i] for g in all_gradients], axis=0)
            avg_weight_gradients.append(avg_weight)
            avg_bias_gradients.append(avg_bias)
        
        avg_gradients = {
            "weights": avg_weight_gradients,
            "biases": avg_bias_gradients
        }
        
        # Apply averaged gradients to the global model
        parameter_server.apply_gradients(avg_gradients, total_compute_time)
        
        # Record metrics
        iteration_time = time.time() - iteration_start
        metrics["iterations"].append(iteration + 1)
        metrics["times"].append(iteration_time)
        
        print(f"Iteration {iteration + 1} completed in {iteration_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\nSynchronous training completed in {total_time:.2f}s")
    
    return metrics

# TODO: IMPLEMENT ASYNC VERSION
# You'll need to implement:
# 1. async def train_asynchronous(parameter_server, workers, num_iterations)
# This should allow workers to operate independently without synchronization barriers

def main():
    """Main function to run the training simulation."""
    print("Distributed Training with Parameter Server Exercise")
    print("================================================")
    
    # Create parameter server
    parameter_server = ParameterServer(LAYER_SIZES)
    
    # Create workers
    workers = [Worker(i, LAYER_SIZES) for i in range(NUM_WORKERS)]
    
    # Run synchronous training
    sync_metrics = train_synchronous(parameter_server, workers, NUM_ITERATIONS)
    
    # TODO: Uncomment to test your async implementation
    # Reset parameter server and workers
    # parameter_server = ParameterServer(LAYER_SIZES)
    # workers = [Worker(i, LAYER_SIZES) for i in range(NUM_WORKERS)]
    # 
    # # Run asynchronous training
    # async_metrics = asyncio.run(train_asynchronous(parameter_server, workers, NUM_ITERATIONS))
    # 
    # # Compare performance
    # sync_total_time = sum(sync_metrics["times"])
    # async_total_time = sum(async_metrics["times"])
    # speedup = sync_total_time / async_total_time
    # 
    # print("\nPerformance Comparison:")
    # print(f"Synchronous training total time: {sync_total_time:.2f}s")
    # print(f"Asynchronous training total time: {async_total_time:.2f}s")
    # print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()