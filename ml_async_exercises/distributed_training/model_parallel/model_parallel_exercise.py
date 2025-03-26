#!/usr/bin/env python3
"""
Model Parallel Training Exercise

This script simulates model-parallel neural network training where:
1. Different layers of a neural network run on different nodes
2. Forward pass requires sending activations from one node to the next
3. Backward pass requires sending gradients back in reverse order
4. Nodes must coordinate to ensure correct execution order

Your task: Implement the async version to enable efficient pipelining of
multiple batches through the network.
"""
import time
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# Configuration
NUM_LAYERS = 3
BATCH_SIZE = 16
INPUT_SIZE = 784  # 28x28 input (like MNIST)
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
NUM_BATCHES = 10
SIMULATE_COMPUTATION = True

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

# TODO: IMPLEMENT ASYNC VERSION
# You'll need to implement:
# 1. Async methods for forward and backward passes
# 2. Pipeline execution where multiple batches can be in-flight simultaneously
# 3. Proper coordination between nodes to maintain execution order
#
# async def train_async(self, num_batches: int) -> List[Dict[str, Any]]:
#     """Train multiple batches asynchronously using pipelining."""
#     # Your implementation here

def main():
    """Main function to run the training simulation."""
    print("Model Parallel Training Exercise")
    print("===============================")
    
    # Create model-parallel network
    network = ModelParallelNetwork()
    
    # Train synchronously
    sync_results = network.train_synchronous(NUM_BATCHES)
    
    # TODO: Uncomment to run your async implementation
    # # Reset network
    # network = ModelParallelNetwork()
    #
    # # Train asynchronously
    # async_results = asyncio.run(network.train_async(NUM_BATCHES))
    #
    # # Compare performance
    # sync_total_time = sum(r["total_time"] for r in sync_results)
    # async_total_time = sum(r["total_time"] for r in async_results)
    # sync_avg_time = sync_total_time / len(sync_results)
    # async_avg_time = async_total_time / len(async_results)
    #
    # print("\nPerformance Comparison:")
    # print(f"Synchronous: Avg {sync_avg_time:.3f}s per batch, total {sync_total_time:.3f}s")
    # print(f"Asynchronous: Avg {async_avg_time:.3f}s per batch, total {async_total_time:.3f}s")
    # print(f"Speedup: {sync_total_time / async_total_time:.2f}x")

if __name__ == "__main__":
    main()