# Learn Python Async Programming

This repository contains practical examples to help you learn Python's asynchronous programming features.

## What is Async Programming?

Asynchronous programming allows your code to perform multiple operations concurrently without using threads. This can dramatically improve performance for I/O-bound tasks (like network requests or file operations).

Key concepts:
- **Coroutines**: Functions defined with `async def` that can pause execution with `await`
- **Event Loop**: The central execution environment that manages and distributes tasks
- **Tasks**: Units of work scheduled in the event loop
- **await**: Expression that yields control back to the event loop while waiting for an operation to complete

## Examples in this Repository

1. **async_basics.py**: Fundamentals of async/await syntax and concurrent vs sequential execution
2. **web_scraper.py**: Real-world example of async HTTP requests
3. **async_patterns.py**: Advanced patterns including cancellation, producer-consumer queues, and concurrency limiting
4. **file_processor.py**: Using async for file I/O operations

## Requirements

- Python 3.7+ (for full asyncio support)
- Additional packages for some examples:
  - `aiohttp` (for web_scraper.py): `pip install aiohttp`
  - `requests` (for comparison in web_scraper.py): `pip install requests`

## Running the Examples

Each Python file is executable. Simply run:

```bash
python async_basics.py
python web_scraper.py  # Requires aiohttp and requests
python async_patterns.py
python file_processor.py
```

## Learning Path

1. Start with **async_basics.py** to understand the core concepts
2. Move on to **web_scraper.py** to see a practical application
3. Explore **async_patterns.py** for more advanced techniques
4. Check out **file_processor.py** to see how to handle blocking I/O operations

## Key Takeaways

- Async is best for I/O-bound tasks, not CPU-bound tasks
- The performance benefits come from doing other work while waiting for I/O
- Not all libraries support async operations - sometimes you need to use run_in_executor
- Proper error handling and task management are crucial in async programming