#!/usr/bin/env python3
"""
Gradient Accumulation with Async Validation - SOLUTION

This script demonstrates an async implementation of gradient accumulation
with concurrent validation during training and early stopping.
"""
import time
import random
import numpy as np
import asyncio
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
        self.lock = asyncio.Lock()  # Lock for model updates
        
        # Initialize random weights and biases
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.biases.append(np.random.randn(layer_sizes[i+1]) * 0.1)
    
    async def apply_gradients(self, gradients: Dict[str, List[np.ndarray]], 
                              learning_rate: float = 0.01) -> None:
        """Apply gradients to update model parameters with thread safety."""
        async with self.lock:
            # Update weights and biases
            weight_gradients = gradients["weights"]
            bias_gradients = gradients["biases"]
            
            for i in range(len(self.weights)):
                self.weights[i] -= learning_rate * weight_gradients[i]
                self.biases[i] -= learning_rate * bias_gradients[i]
            
            self.step += 1
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current model state for checkpointing."""
        async with self.lock:
            return {
                "weights": [w.copy() for w in self.weights],
                "biases": [b.copy() for b in self.biases],
                "step": self.step
            }
    
    async def set_state(self, state: Dict[str, Any]) -> None:
        """Set model state from checkpoint."""
        async with self.lock:
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
    
    async def compute_gradients_async(self) -> Tuple[Dict[str, List[np.ndarray]], float]:
        """
        Compute gradients asynchronously using local data.
        In a real system, this would compute actual gradients via backpropagation.
        
        Returns:
            - Gradients
            - Computation time
        """
        # Simulate computation time - workers are heterogeneous
        compute_time = 0.1 + 0.2 * np.abs(np.sin(self.worker_id * 0.5)) + 0.05 * random.random()
        
        if SIMULATE_COMPUTATION:
            await asyncio.sleep(compute_time)  # Non-blocking sleep to simulate computation
        
        # Generate fake gradients (in a real system, these would be computed via backprop)
        # We need to make a copy of the model weights for gradient computation
        # In a real system, we would have a copy of the model parameters in the worker
        async with self.model.lock:
            model_weights = [w.copy() for w in self.model.weights]
            model_biases = [b.copy() for b in self.model.biases]
            model_step = self.model.step
        
        weight_gradients = []
        bias_gradients = []
        
        for w, b in zip(model_weights, model_biases):
            # Generate random gradients scaled by a decaying factor to simulate convergence
            decay_factor = 1.0 / (1.0 + 0.01 * model_step)
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
        self.lock = asyncio.Lock()
        self.update_event = asyncio.Event()  # Event to signal when an update is completed
        self.update_count = 0
    
    async def add_gradients_async(self, gradients: Dict[str, List[np.ndarray]], compute_time: float) -> bool:
        """
        Add gradients to the accumulator asynchronously.
        
        Returns:
            - True if accumulation is complete and model was updated
            - False if still accumulating
        """
        # Acquire lock to safely modify the accumulated gradients
        async with self.lock:
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
                await self.model.apply_gradients(self.accumulated_gradients)
                
                # Increment update count and signal completion
                self.update_count += 1
                self.update_event.set()
                self.update_event.clear()
                
                print(f"Applied accumulated gradients to model after {self.accumulation_steps} steps "
                      f"(update #{self.update_count})")
                
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
        self.lock = asyncio.Lock()
    
    async def validate_async(self) -> Dict[str, Any]:
        """
        Validate the current model asynchronously and return metrics.
        In a real system, this would evaluate the model on a validation dataset.
        """
        # Simulate validation computation
        validation_time = 0.5 + 0.1 * random.random()
        
        if SIMULATE_COMPUTATION:
            await asyncio.sleep(validation_time)  # Non-blocking sleep
        
        # Get current model state for validation
        model_state = await self.model.get_state()
        model_step = model_state["step"]
        
        # Generate a fake validation loss that improves over time with noise
        base_loss = 2.0 / (1.0 + 0.05 * model_step)
        noise = 0.1 * random.random()
        loss = base_loss + noise
        
        # Update best model state with lock to ensure thread safety
        async with self.lock:
            # Track the best model
            improved = False
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_model_state = model_state
                self.epochs_without_improvement = 0
                improved = True
            else:
                self.epochs_without_improvement += 1
            
            metrics = {
                "step": model_step,
                "loss": loss,
                "best_loss": self.best_loss,
                "improved": improved,
                "epochs_without_improvement": self.epochs_without_improvement,
                "validation_time": validation_time
            }
            
            self.validation_history.append(metrics)
        
        print(f"Validation at step {model_step}: loss={loss:.4f} "
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
        self.stop_event = asyncio.Event()
    
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
                # In sync version, we call the async function with run_until_complete
                loop = asyncio.new_event_loop()
                gradients, compute_time = loop.run_until_complete(worker.compute_gradients_async())
                loop.close()
                
                # Add gradients to accumulator
                loop = asyncio.new_event_loop()
                update_complete = loop.run_until_complete(
                    self.accumulator.add_gradients_async(gradients, compute_time)
                )
                loop.close()
                
                # If we've completed an update, check if we should validate
                if update_complete:
                    update_count += 1
                    
                    # Periodically validate
                    if update_count % self.validate_every_n_steps == 0:
                        loop = asyncio.new_event_loop()
                        metrics = loop.run_until_complete(self.validator.validate_async())
                        loop.close()
                        
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
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.model.set_state(self.validator.best_model_state))
            loop.close()
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
    
    async def worker_loop(self, worker: Worker) -> None:
        """Async worker loop that continuously computes gradients until stopped."""
        while not self.stop_event.is_set():
            # Compute gradients
            gradients, compute_time = await worker.compute_gradients_async()
            
            # Add to accumulator
            await self.accumulator.add_gradients_async(gradients, compute_time)
    
    async def validation_loop(self) -> None:
        """Async validation loop that runs validations periodically."""
        prev_update_count = 0
        
        while not self.stop_event.is_set():
            # Wait for updates to happen
            current_update_count = self.accumulator.update_count
            
            # Check if we should validate
            if current_update_count > 0 and (current_update_count % self.validate_every_n_steps == 0) and (current_update_count != prev_update_count):
                # Run validation
                metrics = await self.validator.validate_async()
                prev_update_count = current_update_count
                
                # Check early stopping condition
                if metrics["epochs_without_improvement"] >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {current_update_count} updates")
                    self.stop_event.set()
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    async def training_monitor(self, num_iterations: int) -> None:
        """Monitor training progress and stop after max iterations."""
        iteration = 0
        
        while not self.stop_event.is_set() and iteration < num_iterations:
            # Record metrics for this iteration
            iteration_start = time.time()
            
            # Wait for the next update event or timeout
            try:
                # Wait for a short period or until an update happens
                await asyncio.wait_for(
                    self.accumulator.update_event.wait(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                # Just continue if no update occurred
                pass
            
            # Check if we should record a new iteration
            current_update_count = self.accumulator.update_count
            iteration_time = time.time() - iteration_start
            
            self.training_history.append({
                "iteration": iteration,
                "update_count": current_update_count,
                "time": iteration_time
            })
            
            print(f"Iteration {iteration + 1} completed")
            iteration += 1
        
        # Signal workers to stop if we've reached max iterations
        if not self.stop_event.is_set():
            print(f"Reached maximum iterations ({num_iterations})")
            self.stop_event.set()
    
    async def train_asynchronous(self, num_iterations: int) -> Dict[str, Any]:
        """
        Train the model asynchronously for a specific number of iterations.
        
        In the asynchronous version:
        1. Multiple workers compute gradients concurrently
        2. Gradients are accumulated as they arrive
        3. When accumulation is complete, the model is updated
        4. Validation runs concurrently with training
        5. Training stops if validation doesn't improve for patience steps
        """
        print("\nStarting ASYNCHRONOUS training...")
        start_time = time.time()
        
        # Clear the stop event
        self.stop_event.clear()
        
        # Start worker tasks
        worker_tasks = [asyncio.create_task(self.worker_loop(worker)) for worker in self.workers]
        
        # Start validation task
        validation_task = asyncio.create_task(self.validation_loop())
        
        # Start training monitor
        monitor_task = asyncio.create_task(self.training_monitor(num_iterations))
        
        # Wait for training to complete (either by max iterations or early stopping)
        await asyncio.gather(monitor_task)
        
        # Cancel all worker and validation tasks
        for task in worker_tasks:
            task.cancel()
        validation_task.cancel()
        
        # Wait for all tasks to complete
        try:
            await asyncio.gather(*worker_tasks, validation_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        
        # Restore best model
        if self.validator.best_model_state is not None:
            await self.model.set_state(self.validator.best_model_state)
            print(f"Restored best model with loss {self.validator.best_loss:.4f}")
        
        total_time = time.time() - start_time
        
        results = {
            "total_time": total_time,
            "iterations": len(self.training_history),
            "updates": self.accumulator.update_count,
            "best_loss": self.validator.best_loss,
            "early_stopped": self.validator.epochs_without_improvement >= self.early_stopping_patience,
            "avg_iteration_time": sum(h["time"] for h in self.training_history) / max(1, len(self.training_history)),
            "validation_history": self.validator.validation_history
        }
        
        print(f"\nAsynchronous training completed in {total_time:.2f}s "
              f"after {len(self.training_history)} iterations and {self.accumulator.update_count} updates")
        print(f"Best validation loss: {self.validator.best_loss:.4f}")
        
        return results

def train_async_wrapper(manager: TrainingManager, num_iterations: int) -> Dict[str, Any]:
    """Wrapper to run the async training in an event loop."""
    return asyncio.run(manager.train_asynchronous(num_iterations))

def main():
    """Main function to run the training simulation."""
    print("Gradient Accumulation with Async Validation Solution")
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
    
    # Reset everything for async training
    model = Model(LAYER_SIZES)
    workers = [Worker(i, model) for i in range(NUM_WORKERS)]
    accumulator = GradientAccumulator(model, GRADIENT_ACCUMULATION_STEPS)
    validator = Validator(model)
    manager = TrainingManager(
        model=model,
        workers=workers,
        accumulator=accumulator,
        validator=validator,
        validate_every_n_steps=VALIDATE_EVERY_N_STEPS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    
    # Train asynchronously
    async_results = train_async_wrapper(manager, NUM_ITERATIONS)
    
    # Compare performance
    print("\nPerformance Comparison:")
    print(f"Synchronous: {sync_results['total_time']:.2f}s, {sync_results['updates']} updates, "
          f"loss={sync_results['best_loss']:.4f}")
    print(f"Asynchronous: {async_results['total_time']:.2f}s, {async_results['updates']} updates, "
          f"loss={async_results['best_loss']:.4f}")
    print(f"Speedup: {sync_results['total_time'] / async_results['total_time']:.2f}x")

if __name__ == "__main__":
    main()