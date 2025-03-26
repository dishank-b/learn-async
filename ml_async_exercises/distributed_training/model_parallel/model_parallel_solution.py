#!/usr/bin/env python3
"""
Model Parallel Training Exercise - SOLUTION

This script demonstrates an async implementation of model-parallel neural network training
with efficient pipelining of multiple batches through the network.
"""
import time
import random
import numpy as np
import asyncio
from typing import Dict, List, Any, Tuple, Optional, Deque
from collections import deque

# Configuration
NUM_LAYERS = 3
BATCH_SIZE = 16
INPUT_SIZE = 784  # 28x28 input (like MNIST)
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
NUM_BATCHES = 10
SIMULATE_COMPUTATION = True
MAX_PIPELINE_DEPTH = 3  # Maximum number of batches in the pipeline simultaneously

class LayerNode:
    """Simulates a compute node that handles one layer of a neural network."""
    
    def __init__(self, node_id: int, input_size: int, output_size: int):
        """Initialize a layer node with random weights."""
        self.node_id = node_id
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights and biases (simulated)
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.random.randn(output_size) * 0.1
        
        # Stats
        self.forward_count = 0
        self.backward_count = 0
        self.compute_time_forward = 0.0
        self.compute_time_backward = 0.0
        
        # Locks for async execution
        self.forward_lock = asyncio.Lock()
        self.backward_lock = asyncio.Lock()

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform forward pass computation.
        In a real system, this would do actual matrix multiplication and activation.
        """
        if not SIMULATE_COMPUTATION:
            # Actual computation (for validation)
            output = np.dot(input_data, self.weights) + self.bias
            return np.maximum(0, output)  # ReLU activation
        
        # Simulate computation time - more inputs or outputs = more time
        compute_time = 0.01 * (self.input_size * self.output_size / 10000) * (1 + 0.2 * random.random())
        time.sleep(compute_time)
        
        # Generate simulated output of the right shape
        output = np.random.randn(input_data.shape[0], self.output_size)
        
        # Track stats
        self.forward_count += 1
        self.compute_time_forward += compute_time
        
        print(f"Node {self.node_id}: Forward pass #{self.forward_count} completed in {compute_time:.3f}s")
        return output
    
    def backward(self, output_gradient: np.ndarray, input_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform backward pass computation.
        In a real system, this would compute actual gradients.
        """
        # Simulate computation time
        compute_time = 0.015 * (self.input_size * self.output_size / 10000) * (1 + 0.2 * random.random())
        time.sleep(compute_time)
        
        # Generate simulated input gradient of the right shape
        input_gradient = np.random.randn(output_gradient.shape[0], self.input_size)
        
        # Track stats
        self.backward_count += 1
        self.compute_time_backward += compute_time
        
        print(f"Node {self.node_id}: Backward pass #{self.backward_count} completed in {compute_time:.3f}s")
        return input_gradient
    
    async def forward_async(self, input_data: np.ndarray, batch_id: int) -> np.ndarray:
        """Async version of forward computation."""
        # Acquire lock to ensure only one forward pass at a time per node
        async with self.forward_lock:
            start_time = time.time()
            print(f"Node {self.node_id}: Starting forward pass for batch {batch_id}")
            
            if not SIMULATE_COMPUTATION:
                # Actual computation (for validation)
                output = np.dot(input_data, self.weights) + self.bias
                result = np.maximum(0, output)  # ReLU activation
            else:
                # Simulate computation with non-blocking sleep
                compute_time = 0.01 * (self.input_size * self.output_size / 10000) * (1 + 0.2 * random.random())
                await asyncio.sleep(compute_time)
                result = np.random.randn(input_data.shape[0], self.output_size)
                
                # Track stats
                self.forward_count += 1
                self.compute_time_forward += compute_time
                
                print(f"Node {self.node_id}: Forward pass #{self.forward_count} "
                      f"for batch {batch_id} completed in {compute_time:.3f}s")
            
            return result
    
    async def backward_async(self, output_gradient: np.ndarray, batch_id: int, 
                            input_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Async version of backward computation."""
        # Acquire lock to ensure only one backward pass at a time per node
        async with self.backward_lock:
            start_time = time.time()
            print(f"Node {self.node_id}: Starting backward pass for batch {batch_id}")
            
            # Simulate computation with non-blocking sleep
            compute_time = 0.015 * (self.input_size * self.output_size / 10000) * (1 + 0.2 * random.random())
            await asyncio.sleep(compute_time)
            
            # Generate simulated input gradient of the right shape
            input_gradient = np.random.randn(output_gradient.shape[0], self.input_size)
            
            # Track stats
            self.backward_count += 1
            self.compute_time_backward += compute_time
            
            print(f"Node {self.node_id}: Backward pass #{self.backward_count} "
                  f"for batch {batch_id} completed in {compute_time:.3f}s")
            
            return input_gradient

class ModelParallelNetwork:
    """Manages the model-parallel neural network training."""
    
    def __init__(self):
        """Initialize the network with layer nodes."""
        self.nodes = []
        
        # Create layer nodes with proper dimensions
        layer_sizes = [INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE]
        for i in range(len(layer_sizes) - 1):
            node = LayerNode(i, layer_sizes[i], layer_sizes[i+1])
            self.nodes.append(node)
        
        self.num_nodes = len(self.nodes)
        
        # For tracking pipeline state
        self.batch_activations = {}
        self.semaphore = asyncio.Semaphore(MAX_PIPELINE_DEPTH)
    
    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a fake data batch."""
        x = np.random.randn(BATCH_SIZE, INPUT_SIZE)
        y = np.random.randint(0, OUTPUT_SIZE, size=BATCH_SIZE)
        return x, y
    
    def train_batch_synchronous(self, batch_id: int) -> Dict[str, Any]:
        """
        Train one batch synchronously through the network.
        
        Steps:
        1. Forward pass: input -> node 0 -> node 1 -> ... -> output
        2. Backward pass: output grad -> node n -> node n-1 -> ... -> input grad
        """
        start_time = time.time()
        print(f"\nProcessing batch {batch_id} synchronously...")
        
        # Generate a data batch
        inputs, targets = self.generate_batch()
        
        # Store activations for the backward pass
        activations = [inputs]
        
        # Forward pass
        current_input = inputs
        for node in self.nodes:
            output = node.forward(current_input)
            activations.append(output)
            current_input = output
        
        final_output = activations[-1]
        
        # Simulate loss computation and initial gradient
        # In a real system, this would compute an actual loss gradient
        output_gradient = np.random.randn(BATCH_SIZE, self.nodes[-1].output_size)
        
        # Backward pass
        current_gradient = output_gradient
        for i in range(len(self.nodes) - 1, -1, -1):
            node = self.nodes[i]
            input_gradient = node.backward(current_gradient, activations[i])
            current_gradient = input_gradient
        
        # Compute statistics
        elapsed = time.time() - start_time
        forward_time = sum(node.compute_time_forward for node in self.nodes)
        backward_time = sum(node.compute_time_backward for node in self.nodes)
        
        stats = {
            "batch_id": batch_id,
            "total_time": elapsed,
            "forward_time": forward_time,
            "backward_time": backward_time
        }
        
        print(f"Batch {batch_id} completed in {elapsed:.3f}s (forward: {forward_time:.3f}s, backward: {backward_time:.3f}s)")
        return stats
    
    def train_synchronous(self, num_batches: int) -> List[Dict[str, Any]]:
        """Train multiple batches synchronously."""
        print("Starting SYNCHRONOUS training...")
        start_time = time.time()
        
        results = []
        for i in range(num_batches):
            result = self.train_batch_synchronous(i)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_batch_time = sum(r["total_time"] for r in results) / len(results)
        
        print(f"\nSynchronous training completed: {num_batches} batches in {total_time:.3f}s")
        print(f"Average batch time: {avg_batch_time:.3f}s")
        
        return results
    
    async def process_batch_async(self, batch_id: int) -> Dict[str, Any]:
        """
        Process a single batch asynchronously through the pipelined network.
        """
        # Acquire semaphore to limit pipeline depth
        async with self.semaphore:
            start_time = time.time()
            print(f"\nProcessing batch {batch_id} asynchronously...")
            
            # Generate a data batch
            inputs, targets = self.generate_batch()
            
            # Store activations for the backward pass
            activations = [inputs]
            self.batch_activations[batch_id] = activations
            
            # Forward pass
            current_input = inputs
            forward_start = time.time()
            
            for i, node in enumerate(self.nodes):
                output = await node.forward_async(current_input, batch_id)
                activations.append(output)
                current_input = output
                
            forward_time = time.time() - forward_start
            
            # Simulate loss computation and initial gradient
            output_gradient = np.random.randn(BATCH_SIZE, self.nodes[-1].output_size)
            
            # Backward pass
            backward_start = time.time()
            current_gradient = output_gradient
            
            for i in range(len(self.nodes) - 1, -1, -1):
                node = self.nodes[i]
                input_gradient = await node.backward_async(current_gradient, batch_id, activations[i])
                current_gradient = input_gradient
                
            backward_time = time.time() - backward_start
            
            # Clean up
            del self.batch_activations[batch_id]
            
            # Compute statistics
            elapsed = time.time() - start_time
            
            stats = {
                "batch_id": batch_id,
                "total_time": elapsed,
                "forward_time": forward_time,
                "backward_time": backward_time
            }
            
            print(f"Batch {batch_id} completed in {elapsed:.3f}s "
                  f"(forward: {forward_time:.3f}s, backward: {backward_time:.3f}s)")
            return stats
    
    async def train_async(self, num_batches: int) -> List[Dict[str, Any]]:
        """
        Train multiple batches asynchronously using pipelining.
        
        This approach allows multiple batches to flow through the network
        simultaneously, with each node processing different batches in parallel.
        """
        print("Starting ASYNCHRONOUS training...")
        global_start = time.time()
        
        # Create a task for each batch
        tasks = [self.process_batch_async(i) for i in range(num_batches)]
        
        # Wait for all batches to complete
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - global_start
        avg_batch_time = sum(r["total_time"] for r in results) / len(results)
        
        print(f"\nAsynchronous training completed: {num_batches} batches in {total_time:.3f}s")
        print(f"Average batch time: {avg_batch_time:.3f}s")
        print(f"Pipeline efficiency: {sum(r['total_time'] for r in results) / total_time:.2f}x")
        
        return results

def main():
    """Main function to run the training simulation."""
    print("Model Parallel Training Solution")
    print("==============================")
    
    # Create model-parallel network
    network = ModelParallelNetwork()
    
    # Train synchronously
    sync_results = network.train_synchronous(NUM_BATCHES)
    
    # Reset network
    network = ModelParallelNetwork()
    
    # Train asynchronously
    import asyncio
    async_results = asyncio.run(network.train_async(NUM_BATCHES))
    
    # Compare performance
    sync_total_time = sum(r["total_time"] for r in sync_results)
    async_total_time = sum(r["total_time"] for r in async_results)
    sync_avg_time = sync_total_time / len(sync_results)
    async_avg_time = async_total_time / len(async_results)
    
    print("\nPerformance Comparison:")
    print(f"Synchronous: Avg {sync_avg_time:.3f}s per batch, total {sync_total_time:.3f}s")
    print(f"Asynchronous: Avg {async_avg_time:.3f}s per batch, total {async_total_time:.3f}s")
    print(f"Individual batch time speedup: {sync_avg_time / async_avg_time:.2f}x")
    
    # The key metric is total wall time, which should be significantly less
    # due to pipelining multiple batches
    async_wall_time = time.time() - time.time()  # This will be set during execution
    sync_wall_time = sync_total_time  # For sync, wall time equals sum of batch times
    
    # Alternatively, we can compare against the first/last timestamps
    first_sync = min(r["batch_id"] for r in sync_results)
    last_sync = max(r["batch_id"] for r in sync_results)
    first_async = min(r["batch_id"] for r in async_results)
    last_async = max(r["batch_id"] for r in async_results)
    
    print(f"Pipeline speedup: {sync_total_time / async_total_time:.2f}x")

if __name__ == "__main__":
    main()