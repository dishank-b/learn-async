# Advanced Async Patterns in Python

## 1. Task Creation and Cancellation

**What it is**: Creating and managing independent units of work that can be scheduled, monitored, and cancelled.

**How it works**:
- `asyncio.create_task()` wraps a coroutine into a Task that schedules execution on the event loop
- `task.cancel()` requests cancellation, raising a `CancelledError` inside the task
- Tasks must properly handle `CancelledError` for clean cancellation

**Key code**:
```python
task = asyncio.create_task(cancellable_task("C", 10.0))
await asyncio.sleep(3)  # Let it run for a bit
task.cancel()  # Request cancellation
```

The cancellable_task handles cancellation properly by:
```python
try:
    # Do work
except asyncio.CancelledError:
    print(f"Task {name} was cancelled after {time.time() - start:.1f}s")
    raise  # Re-raising is important for proper cancellation
```

**Real-world uses**: Timeouts, user interruptions, graceful shutdowns

## 2. Producer-Consumer Pattern with Queue

**What it is**: A pattern where some tasks produce data and others consume it, with a queue mediating between them.

**How it works**:
- `asyncio.Queue` provides an async-compatible queue with backpressure
- Producers call `await queue.put(item)` to add items
- Consumers call `await queue.get()` to retrieve items
- `queue.task_done()` signals completion of an item

**Key code**:
```python
# Producer puts items in queue
for i in range(count):
    await queue.put(item)  # Will wait if queue.maxsize is reached

# Consumer processes items
while True:
    item = await queue.get()
    if item is None:  # Termination signal
        queue.task_done()
        break
    # Process item
    queue.task_done()
```

**Real-world uses**: Work distribution, data pipeline processing, job queues

## 3. Semaphores for Concurrency Limiting

**What it is**: A mechanism to limit how many tasks can perform a certain operation simultaneously.

**How it works**:
- `asyncio.Semaphore(n)` creates a counter with maximum value n
- `async with semaphore:` acquires the semaphore (waiting if necessary) and releases it when the block completes
- Only n tasks can be inside the semaphore block at once

**Key code**:
```python
semaphore = asyncio.Semaphore(3)  # Allow only 3 concurrent operations

async def limited_concurrent_task(semaphore, task_id):
    async with semaphore:  # Wait if already 3 tasks are running
        # Do limited work here...
```

**Real-world uses**: Rate limiting, database connection pooling, resource management

## 4. Handling Blocking I/O with Executors

**What it is**: Running blocking I/O operations in a thread pool to avoid blocking the event loop.

**How it works**:
- `loop.run_in_executor()` runs a function in a thread or process pool
- Frees the event loop to handle other tasks while I/O is in progress
- Returns a Future that completes when the function finishes

**Key code**:
```python
# Move blocking file I/O to a thread pool
content = await loop.run_in_executor(None, lambda: open(filename, "rb").read())
```

**Real-world uses**: File I/O, CPU-bound tasks, interfacing with non-async libraries

## 5. Using asyncio.gather for Concurrent Operations

**What it is**: Running multiple coroutines concurrently and collecting all their results.

**How it works**:
- `asyncio.gather(*tasks)` runs all tasks concurrently
- Waits for all tasks to complete (or fail)
- Returns a list of results in the same order as inputs
- With `return_exceptions=True`, returns exceptions instead of raising them

**Key code**:
```python
# Process files asynchronously with concurrent execution
tasks = [process_file_async(filepath, loop) for filepath in files]
results = await asyncio.gather(*tasks)
```

**Real-world uses**: Parallel API calls, concurrent processing, fan-out/fan-in workflows

Each of these patterns provides powerful tools for structuring async applications efficiently, handling resources properly, and managing concurrency in a controlled way.