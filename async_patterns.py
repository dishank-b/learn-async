#!/usr/bin/env python3
"""
Advanced async patterns in Python.
Demonstrates several common async patterns and techniques.
"""
import asyncio
import random
import time
from typing import List, Dict, Any

# =============================================
# Pattern 1: Task creation and cancellation
# =============================================

async def cancellable_task(name: str, duration: float):
    """A task that can be cancelled."""
    print(f"Task {name} started, will run for {duration:.1f}s")
    try:
        start = time.time()
        while time.time() - start < duration:
            # Do some work and periodically yield control
            await asyncio.sleep(0.1)
            print(f"Task {name} running... {(time.time() - start):.1f}s elapsed")
    except asyncio.CancelledError:
        print(f"Task {name} was cancelled after {time.time() - start:.1f}s")
        raise  # Re-raise to properly mark the task as cancelled
    else:
        print(f"Task {name} completed successfully")
        return f"Result from {name}"

async def demo_task_cancellation():
    """Demonstrate creating and cancelling tasks."""
    print("\n=== Task Cancellation Demo ===")
    
    # Create three tasks with different durations
    task1 = asyncio.create_task(cancellable_task("A", 2.0))
    task2 = asyncio.create_task(cancellable_task("B", 5.0))
    task3 = asyncio.create_task(cancellable_task("C", 10.0))
    
    # Let them run for a bit
    await asyncio.sleep(3)
    
    # Cancel the longest-running task
    print("Cancelling task C...")
    task3.cancel()
    
    # Wait for all tasks to complete or be cancelled
    try:
        results = await asyncio.gather(task1, task2, task3, return_exceptions=True)
        for i, result in enumerate(results, 1):
            if isinstance(result, asyncio.CancelledError):
                print(f"Task {i} confirmed cancelled")
            else:
                print(f"Task {i} result: {result}")
    except Exception as e:
        print(f"Error waiting for tasks: {e}")

# =============================================
# Pattern 2: Producer-Consumer with Queue
# =============================================

async def producer(queue: asyncio.Queue, name: str, count: int):
    """Produces items and puts them in the queue."""
    for i in range(count):
        item = f"Item {i} from {name}"
        await queue.put(item)
        print(f"Producer {name} added: {item}")
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Random delay
    
    # Signal this producer is done
    await queue.put(None)
    print(f"Producer {name} finished")

async def consumer(queue: asyncio.Queue, name: str):
    """Consumes items from the queue."""
    while True:
        item = await queue.get()
        if item is None:
            # Signal to stop
            queue.task_done()
            break
            
        # Process the item
        print(f"Consumer {name} processing: {item}")
        await asyncio.sleep(random.uniform(0.2, 0.8))  # Simulate processing time
        queue.task_done()
    
    print(f"Consumer {name} finished")

async def demo_producer_consumer():
    """Demonstrate the producer-consumer pattern using a queue."""
    print("\n=== Producer-Consumer Demo ===")
    queue = asyncio.Queue(maxsize=5)  # Limit queue size to demonstrate backpressure
    
    # Create producers and consumers
    producers = [
        asyncio.create_task(producer(queue, f"P{i}", random.randint(3, 7)))
        for i in range(2)
    ]
    
    consumers = [
        asyncio.create_task(consumer(queue, f"C{i}"))
        for i in range(3)
    ]
    
    # Wait for producers to finish
    await asyncio.gather(*producers)
    
    # Add termination signals for consumers
    for _ in range(len(consumers)):
        await queue.put(None)
    
    # Wait for consumers to finish
    await asyncio.gather(*consumers)
    
    print("Producer-Consumer demo completed")

# =============================================
# Pattern 3: Semaphore for limiting concurrency
# =============================================

async def limited_concurrent_task(semaphore: asyncio.Semaphore, task_id: int):
    """Task that uses a semaphore to limit concurrency."""
    async with semaphore:
        print(f"Task {task_id} acquired semaphore, starting work")
        await asyncio.sleep(random.uniform(0.5, 2.0))  # Simulate work
        print(f"Task {task_id} completed work, releasing semaphore")
        return f"Result from task {task_id}"

async def demo_concurrency_limit():
    """Demonstrate limiting concurrency with a semaphore."""
    print("\n=== Concurrency Limiting Demo ===")
    # Allow only 3 tasks to run concurrently
    semaphore = asyncio.Semaphore(3)
    
    # Create 10 tasks
    tasks = [
        limited_concurrent_task(semaphore, i)
        for i in range(10)
    ]
    
    # Run all tasks and collect results
    results = await asyncio.gather(*tasks)
    print(f"All limited concurrency tasks completed with results: {results}")

# =============================================
# Main function to run all demos
# =============================================

async def main():
    print("ADVANCED ASYNC PATTERNS DEMONSTRATION")
    
    # Run each demo
    await demo_task_cancellation()
    await demo_producer_consumer()
    await demo_concurrency_limit()
    
    print("\nAll demos completed!")

if __name__ == "__main__":
    asyncio.run(main())