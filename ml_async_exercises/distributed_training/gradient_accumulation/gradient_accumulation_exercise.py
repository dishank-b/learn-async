#!/usr/bin/env python3
"""
Gradient Accumulation with Async Validation Exercise

This script simulates a training system where:
1. Multiple workers compute gradients in parallel
2. Gradients are accumulated for larger effective batch sizes
3. Periodic validation happens concurrently with training
4. Early stopping is implemented based on validation metrics

Your task: Convert the synchronous implementation to async for better
efficiency and overlapping validation with training.
"""
import time
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# Constants
NUM_WORKERS = 4
NUM_ITERATIONS = 50
GRADIENT_ACCUMULATION_STEPS = 8
VALIDATE_EVERY_N_STEPS = 5
EARLY_STOPPING_PATIENCE = 3
LAYER_SIZES = [784, 128, 10]  # Input -> Hidden -> Output
SIMULATE_COMPUTATION = True

class Model:
    """Simulated neural network model for training."""
    
    def __init__(self, layer_sizes: List[int]):
        """Initialize model with random weights."""
        self.weights = []
        self.biases = []
        self.layer_sizes = layer_sizes
        self.step = 0
        
        # Initialize random weights and biases
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.biases.append(np.random.randn(layer_sizes[i+1]) * 0.1)
    
    def apply_gradients(self, gradients: Dict[str, List[np.ndarray]], 
                        learning_rate: float = 0.01) -> None:
        """Apply gradients to update model parameters."""
        # Update weights and biases
        weight_gradients = gradients["weights"]
        bias_gradients = gradients["biases"]
        
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]
        
        self.step += 1
        
    def get_state(self) -> Dict[str, Any]:
        """Get current model state for checkpointing."""
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
            "step": self.step
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set model state from checkpoint."""
        self.weights = [w.copy() for w in state["weights"]]
        self.biases = [b.copy() for b in state["biases"]]
        self.step = state["step"]

class Worker:
    """Worker that computes gradients on data batches."""
    
    def __init__(self, worker_id: int, model: Model):
        """Initialize worker with an ID and reference to the model."""
        self.worker_id = worker_id
        self.model = model
        self.compute_time_history = []
    
    def compute_gradients(self) -> Tuple[Dict[str, List[np.ndarray]], float]:
        """
        Compute gradients using local data.
        In a real system, this would compute actual gradients via backpropagation.
        
        Returns:
            - Gradients
            - Computation time
        """
        # Simulate computation time - workers are heterogeneous
        compute_time = 0.1 + 0.2 * np.abs(np.sin(self.worker_id * 0.5)) + 0.05 * random.random()
        
        if SIMULATE_COMPUTATION:
            time.sleep(compute_time)  # Simulate computation time
        
        # Generate fake gradients (in a real system, these would be computed via backprop)
        weight_gradients = []
        bias_gradients = []
        
        for w, b in zip(self.model.weights, self.model.biases):
            # Generate random gradients scaled by a decaying factor to simulate convergence
            decay_factor = 1.0 / (1.0 + 0.01 * self.model.step)
            weight_gradients.append(np.random.randn(*w.shape) * 0.1 * decay_factor)
            bias_gradients.append(np.random.randn(*b.shape) * 0.1 * decay_factor)
        
        gradients = {
            "weights": weight_gradients,
            "biases": bias_gradients
        }
        
        self.compute_time_history.append(compute_time)
        
        print(f"Worker {self.worker_id}: Computed gradients (compute time: {compute_time:.2f}s)")
        
        return gradients, compute_time

class GradientAccumulator:
    """Accumulates gradients from multiple workers."""
    
    def __init__(self, model: Model, accumulation_steps: int):
        """Initialize the accumulator."""
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.accumulated_gradients = None
        self.current_step = 0
        self.total_compute_time = 0.0
    
    def add_gradients(self, gradients: Dict[str, List[np.ndarray]], compute_time: float) -> bool:
        """
        Add gradients to the accumulator.
        
        Returns:
            - True if accumulation is complete and model was updated
            - False if still accumulating
        """
        # Initialize accumulated gradients if this is the first batch
        if self.accumulated_gradients is None:
            self.accumulated_gradients = {
                "weights": [np.zeros_like(w) for w in gradients["weights"]],
                "biases": [np.zeros_like(b) for b in gradients["biases"]]
            }
        
        # Add the new gradients
        for i in range(len(gradients["weights"])):
            self.accumulated_gradients["weights"][i] += gradients["weights"][i]
            self.accumulated_gradients["biases"][i] += gradients["biases"][i]
        
        # Track steps and compute time
        self.current_step += 1
        self.total_compute_time += compute_time
        
        print(f"Accumulated gradients: {self.current_step}/{self.accumulation_steps}")
        
        # If we've reached the target number of steps, update the model
        if self.current_step >= self.accumulation_steps:
            # Average the gradients
            for i in range(len(self.accumulated_gradients["weights"])):
                self.accumulated_gradients["weights"][i] /= self.accumulation_steps
                self.accumulated_gradients["biases"][i] /= self.accumulation_steps
            
            # Apply gradients to the model
            self.model.apply_gradients(self.accumulated_gradients)
            
            print(f"Applied accumulated gradients to model after {self.accumulation_steps} steps")
            
            # Reset accumulator
            self.accumulated_gradients = None
            self.current_step = 0
            total_compute_time = self.total_compute_time
            self.total_compute_time = 0.0
            
            return True
        
        return False

class Validator:
    """Validates model performance on a validation dataset."""
    
    def __init__(self, model: Model):
        """Initialize the validator."""
        self.model = model
        self.best_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.validation_history = []
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the current model and return metrics.
        In a real system, this would evaluate the model on a validation dataset.
        """
        # Simulate validation computation
        validation_time = 0.5 + 0.1 * random.random()
        
        if SIMULATE_COMPUTATION:
            time.sleep(validation_time)  # Simulate validation time
        
        # Generate a fake validation loss that improves over time with noise
        # In a real system, this would be computed on a validation dataset
        base_loss = 2.0 / (1.0 + 0.05 * self.model.step)
        noise = 0.1 * random.random()
        loss = base_loss + noise
        
        # Track the best model
        improved = False
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model_state = self.model.get_state()
            self.epochs_without_improvement = 0
            improved = True
        else:
            self.epochs_without_improvement += 1
        
        metrics = {
            "step": self.model.step,
            "loss": loss,
            "best_loss": self.best_loss,
            "improved": improved,
            "epochs_without_improvement": self.epochs_without_improvement,
            "validation_time": validation_time
        }
        
        self.validation_history.append(metrics)
        
        print(f"Validation at step {self.model.step}: loss={loss:.4f} "
              f"(best={self.best_loss:.4f}, epochs_without_improvement={self.epochs_without_improvement})")
        
        return metrics

class TrainingManager:
    """Manages the overall training process."""
    
    def __init__(self, model: Model, workers: List[Worker], 
                accumulator: GradientAccumulator, validator: Validator,
                validate_every_n_steps: int, early_stopping_patience: int):
        """Initialize the training manager."""
        self.model = model
        self.workers = workers
        self.accumulator = accumulator
        self.validator = validator
        self.validate_every_n_steps = validate_every_n_steps
        self.early_stopping_patience = early_stopping_patience
        self.training_history = []
    
    def train_synchronous(self, num_iterations: int) -> Dict[str, Any]:
        """
        Train the model synchronously for a specific number of iterations.
        
        In the synchronous version:
        1. Workers compute gradients one at a time
        2. Gradients are accumulated
        3. When accumulation is complete, the model is updated
        4. Validation runs after the specified number of updates
        5. Training stops if validation doesn't improve for patience steps
        """
        print("\nStarting SYNCHRONOUS training...")
        start_time = time.time()
        
        iteration = 0
        update_count = 0
        should_stop = False
        
        while iteration < num_iterations and not should_stop:
            iteration_start = time.time()
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            # Collect gradients from each worker sequentially
            for worker in self.workers:
                gradients, compute_time = worker.compute_gradients()
                
                # Add gradients to accumulator
                update_complete = self.accumulator.add_gradients(gradients, compute_time)
                
                # If we've completed an update, check if we should validate
                if update_complete:
                    update_count += 1
                    
                    # Periodically validate
                    if update_count % self.validate_every_n_steps == 0:
                        metrics = self.validator.validate()
                        
                        # Check early stopping condition
                        if metrics["epochs_without_improvement"] >= self.early_stopping_patience:
                            print(f"Early stopping triggered after {update_count} updates")
                            should_stop = True
                            break
            
            # Record metrics for this iteration
            iteration_time = time.time() - iteration_start
            self.training_history.append({
                "iteration": iteration,
                "update_count": update_count,
                "time": iteration_time
            })
            
            print(f"Iteration {iteration + 1} completed in {iteration_time:.2f}s")
            iteration += 1
        
        # Restore best model
        if self.validator.best_model_state is not None:
            self.model.set_state(self.validator.best_model_state)
            print(f"Restored best model with loss {self.validator.best_loss:.4f}")
        
        total_time = time.time() - start_time
        
        results = {
            "total_time": total_time,
            "iterations": iteration,
            "updates": update_count,
            "best_loss": self.validator.best_loss,
            "early_stopped": should_stop and iteration < num_iterations,
            "avg_iteration_time": sum(h["time"] for h in self.training_history) / len(self.training_history),
            "validation_history": self.validator.validation_history
        }
        
        print(f"\nSynchronous training completed in {total_time:.2f}s after {iteration} iterations")
        print(f"Best validation loss: {self.validator.best_loss:.4f}")
        
        return results

# TODO: IMPLEMENT ASYNC VERSION
# You'll need to implement:
# 1. async def train_asynchronous(self, num_iterations)
#    - Run workers concurrently
#    - Accumulate gradients asynchronously
#    - Perform validation concurrently with training
#    - Implement early stopping
#
# def train_async_wrapper(manager, num_iterations):
#     """Wrapper to run the async training in an event loop."""
#     import asyncio
#     return asyncio.run(manager.train_asynchronous(num_iterations))

def main():
    """Main function to run the training simulation."""
    print("Gradient Accumulation with Async Validation Exercise")
    print("==================================================")
    
    # Create the model
    model = Model(LAYER_SIZES)
    
    # Create workers
    workers = [Worker(i, model) for i in range(NUM_WORKERS)]
    
    # Create gradient accumulator
    accumulator = GradientAccumulator(model, GRADIENT_ACCUMULATION_STEPS)
    
    # Create validator
    validator = Validator(model)
    
    # Create training manager
    manager = TrainingManager(
        model=model,
        workers=workers,
        accumulator=accumulator,
        validator=validator,
        validate_every_n_steps=VALIDATE_EVERY_N_STEPS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    
    # Train synchronously
    sync_results = manager.train_synchronous(NUM_ITERATIONS)
    
    # TODO: Uncomment to test your async implementation
    # # Reset everything for async training
    # model = Model(LAYER_SIZES)
    # workers = [Worker(i, model) for i in range(NUM_WORKERS)]
    # accumulator = GradientAccumulator(model, GRADIENT_ACCUMULATION_STEPS)
    # validator = Validator(model)
    # manager = TrainingManager(
    #     model=model,
    #     workers=workers,
    #     accumulator=accumulator,
    #     validator=validator,
    #     validate_every_n_steps=VALIDATE_EVERY_N_STEPS,
    #     early_stopping_patience=EARLY_STOPPING_PATIENCE
    # )
    #
    # # Train asynchronously
    # async_results = train_async_wrapper(manager, NUM_ITERATIONS)
    #
    # # Compare performance
    # print("\nPerformance Comparison:")
    # print(f"Synchronous: {sync_results['total_time']:.2f}s, {sync_results['updates']} updates, "
    #       f"loss={sync_results['best_loss']:.4f}")
    # print(f"Asynchronous: {async_results['total_time']:.2f}s, {async_results['updates']} updates, "
    #       f"loss={async_results['best_loss']:.4f}")
    # print(f"Speedup: {sync_results['total_time'] / async_results['total_time']:.2f}x")

if __name__ == "__main__":
    main()