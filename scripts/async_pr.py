import threading
import asyncio
import time

# Threaded class generating values periodically
class PeriodicGeneratorThread:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.generate_values)
        self.thread.start()

    def generate_values(self):
        while not self.stopped.is_set():
            with self.lock:
                self.value += 1

            # Simulate generating a value every second
            time.sleep(1)

    def stop(self):
        self.stopped.set()
        self.thread.join()

# asyncio proxy class for the threaded class
class AsyncGeneratorProxy:
    def __init__(self):
        self.periodic_generator = PeriodicGeneratorThread()
        self.queue = asyncio.Queue()

    async def async_generator(self):
        while True:
            value = await self.queue.get()
            yield value

    async def run_periodic_generator(self):
        while True:
            with self.periodic_generator.lock:
                value = self.periodic_generator.value
            await self.queue.put(value)
            await asyncio.sleep(0.1)  # Adjust this interval as needed

    async def stop(self):
        self.periodic_generator.stop()

# Example usage:

async def main():
    async_proxy = AsyncGeneratorProxy()

    # Start the periodic generator in a separate task
    asyncio.create_task(async_proxy.run_periodic_generator())

    # Get async generator from the proxy
    async_gen = async_proxy.async_generator()

    # Consume values from the async generator
    for _ in range(10):
        value = await async_gen.__anext__()
        print(f"Received value: {value}")

    # Stop the generator
    await async_proxy.stop()

# Run the event loop
asyncio.run(main())
