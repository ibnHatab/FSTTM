
import asyncio
from random import random

class AsyncIterator():
    # constructor, define some state
    def __init__(self):
        self.counter = 0

    # create an instance of the iterator
    def __aiter__(self):
        return self

    # return the next awaitable
    async def __anext__(self):
        # check for no further items
        if self.counter >= 10:
            raise StopAsyncIteration
        # increment the counter
        self.counter += 1
        # return the counter value
        return self.counter

it = AsyncIterator()

awaitable = it.__anext__()
# execute the one step of the iterator and get the result
result = await awaitable

all = [i async for i in AsyncIterator()]