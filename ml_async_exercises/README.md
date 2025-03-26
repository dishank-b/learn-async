# Async Programming Exercises for ML Engineers

This repository contains a set of practical async programming exercises specifically designed for machine learning and AI engineering scenarios. Each exercise simulates a real-world ML engineering task that can benefit from async programming techniques.

## Exercise Structure

Each exercise includes:

1. **Exercise File**: Contains the problem statement and a synchronous implementation
2. **Solution File**: Shows a complete async implementation 
3. **Interview Guide**: Provides context, hints, evaluation criteria, and discussion points

## Exercises

1. **Batch Inference**  
   Optimize a system that performs inference with multiple ML models in parallel

2. **Data Pipeline**  
   Build an ETL pipeline that efficiently downloads and processes data from multiple sources

3. **Model Serving**  
   Create a high-performance model serving system that handles concurrent inference requests

4. **Hyperparameter Tuning**  
   Implement a parallel hyperparameter search system with efficient resource utilization

5. **Distributed Training**  
   Design a coordinator for distributed model training with async communication

## How to Use These Exercises

1. **Learning**: Read through the exercise files to understand common ML async patterns
2. **Practice**: Implement the async version without looking at the solution
3. **Mock Interview**: Have a colleague use the interview guide to simulate a real interview
4. **Self-Assessment**: Compare your solution with the provided implementation

## Key Async Patterns for ML

These exercises demonstrate several important patterns:

- Concurrent API requests
- Producer-consumer with queues
- Resource pooling and management
- Throttling and backpressure
- Coordinating parallel workers
- Efficient I/O operations
- CPU-bound processing in thread pools

## Requirements

To run these exercises, you need:

- Python 3.7+
- asyncio (standard library)
- aiohttp
- aiofiles (for some exercises)

Install dependencies with:
```bash
pip install aiohttp aiofiles
```