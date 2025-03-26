#!/usr/bin/env python3
"""
Basic examples of Python async/await functionality.
This file demonstrates the fundamentals of async programming in Python.
"""
import asyncio
import time

# Basic async function
async def hello_world():
    """Simple async function that returns a greeting."""
    return "Hello, async world!"

# Async function with delay
async def delayed_greeting(name, delay):
    """Async function that simulates work with a delay."""
    print(f"Preparing greeting for {name}...")
    await asyncio.sleep(delay)  # Non-blocking sleep
    return f"Hello, {name}!"

# Multiple coroutines running concurrently
async def main_concurrent():
    """Run multiple coroutines concurrently using gather."""
    print("Starting concurrent execution...")
    start = time.time()
    
    # Create coroutines but don't start them yet
    task1 = delayed_greeting("Alice", 2)
    task2 = delayed_greeting("Bob", 1)
    task3 = delayed_greeting("Charlie", 3)
    
    # Run all tasks concurrently and wait for all to complete
    results = await asyncio.gather(task1, task2, task3)
    
    end = time.time()
    print(f"All greetings completed in {end - start:.2f} seconds")
    print(f"Results: {results}")

# Running coroutines sequentially for comparison
async def main_sequential():
    """Run multiple coroutines sequentially for comparison."""
    print("\nStarting sequential execution...")
    start = time.time()
    
    # Run each task and wait for it to complete before starting the next
    result1 = await delayed_greeting("Alice", 2)
    result2 = await delayed_greeting("Bob", 1)
    result3 = await delayed_greeting("Charlie", 3)
    
    end = time.time()
    print(f"All greetings completed in {end - start:.2f} seconds")
    print(f"Results: [{result1}, {result2}, {result3}]")

# Demonstration
if __name__ == "__main__":
    print("ASYNC BASICS DEMONSTRATION\n")
    
    # Get the event loop
    loop = asyncio.get_event_loop()
    
    # Run the hello_world coroutine
    result = loop.run_until_complete(hello_world())
    print(f"Basic async result: {result}\n")
    
    # Run the concurrent demo
    loop.run_until_complete(main_concurrent())
    
    # Run the sequential demo for comparison
    loop.run_until_complete(main_sequential())
    
    print("\nNote: The concurrent execution is much faster than sequential execution")
    print("even though the total work (sum of all delays) is the same!")