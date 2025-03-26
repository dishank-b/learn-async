# ML Async Programming Interview Guide

## Exercise Overview

This interview exercise focuses on optimizing an ML inference pipeline that makes API calls to multiple model endpoints. The candidate is given a synchronous implementation and asked to optimize it using async techniques.

**Scenario**: The application performs batch inference by sending inputs to multiple ML models hosted as APIs. Each API call takes time, and the goal is to reduce the overall processing time by making the calls concurrently using async programming.

**Time allocation**: ~15 minutes

## Interview Script

### Introduction (1-2 minutes)

"Today, I'd like to see how you'd optimize a common machine learning operations scenario. We have a Python script that performs batch inference by sending requests to multiple ML model APIs. Currently, it's making all API calls sequentially, which is slow.

Your task is to reimplement this using async programming to make it faster. I've provided a synchronous implementation to get you started. You can use any async libraries you're comfortable with, but I recommend `asyncio` and `aiohttp`.

Note that for simplicity, the API endpoints are simulated with delays – you don't need to worry about the actual ML models."

### Coding Phase (10 minutes)

1. Let the candidate review the code for 1-2 minutes
2. Ask them to explain their approach before they start coding
3. Watch their implementation, noting:
   - Do they understand how to structure async functions?
   - Can they use asyncio and aiohttp correctly?
   - Do they recognize the two levels of concurrency (multiple models per item, multiple items in batch)?

### Guiding Questions & Hints

If the candidate gets stuck, use these progressive hints:

1. First hint: "What libraries would you use for async HTTP requests?"
   - Expected answer: aiohttp for async HTTP requests, asyncio for general async operations

2. Second hint: "Think about the different concurrent operations we need. There are actually two levels of concurrency possible here."
   - Expected answer: We need concurrency across different model API calls for a single input, and across different input items in the batch

3. Third hint: "Remember that with async code, we need to create a ClientSession to manage connections efficiently."
   - Expected answer: Create an aiohttp.ClientSession to be shared across requests

4. Fourth hint: "How would you handle waiting for multiple async operations to complete together?"
   - Expected answer: Using asyncio.gather to wait for multiple coroutines

### Solution Walkthrough

When they've finished (or if time is up), walk through the solution:

1. **Basic structure** required:
   - An async function for a single API call
   - An async function to process a single input item across all models
   - An async function to process all input items in the batch
   - A synchronous wrapper to run the async code

2. **Key points** to highlight:
   - Using `asyncio.gather()` for concurrently waiting on multiple operations
   - Creating a shared `aiohttp.ClientSession` for connection pooling
   - Proper error handling in async context
   - Understanding that this makes concurrent requests, not parallel (still single-threaded)

3. **Expected performance**: The async implementation should be ~3-5x faster than the synchronous version because it makes API calls concurrently rather than sequentially.

### Evaluation Criteria

Strong candidates will:
- Demonstrate understanding of async/await syntax and concepts
- Implement proper error handling
- Recognize the two levels of concurrency opportunity
- Explain their code and tradeoffs as they work
- Complete the solution within time constraints
- Discuss potential issues or improvements

### Follow-up Questions (3 minutes)

1. "How would your approach change if some model APIs were unreliable or had long timeouts?"
   - Good answer: Implement timeouts with `asyncio.wait_for` and fallbacks for failed models

2. "What are the limitations of this async approach? When might it not be the best solution?"
   - Good answer: It's still single-threaded so CPU-bound tasks won't benefit; very high volume might require distributed processing

3. "How would you monitor the performance of this system in production?"
   - Good answer: Track latencies per model and overall, set up alerting for degraded performance

## Common Applications of Async in ML Systems

Share these after the exercise:

1. **Distributed Training Coordination**: Async for managing model parameter updates and synchronization
2. **Feature Store Access**: Retrieving features from multiple sources concurrently
3. **Model A/B Testing**: Calling multiple model variants concurrently for comparison
4. **Real-time ML Pipelines**: Handling streaming data while maintaining low latency
5. **Hyperparameter Optimization**: Managing multiple concurrent training runs
6. **ETL Operations**: Downloading and preprocessing data from multiple sources
7. **Online Evaluation**: Gathering metrics from multiple sources for model monitoring
8. **Multi-modal Models**: Fetching results from specialized models (text, vision, audio) in parallel

This exercise tests fundamental async patterns that apply across these and many other ML engineering scenarios.