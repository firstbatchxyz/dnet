"""Asynchronous iteration utilities for dnet."""

import asyncio
from typing import Any, AsyncIterator, Self, Union, final, overload


@final
class arange:
    """Asynchronous equivalent of Python's built-in `range`.

    Supports start, stop, and step, with async iteration via `async for`,
    indexing, slicing, and reversed iteration.

    Example:
        >>> async for i in arange(5):
        ...     print(i)
        0
        1
        2
        3
        4

        >>> a = arange(2, 10, 2)
        >>> print(await a.__anext__())  # first element asynchronously
        2
    """

    __slots__ = ("_start", "_stop", "_step", "_current", "_length")

    def __init__(self, start: int, stop: int | None = None, step: int = 1) -> None:
        """Initialize async range.

        Args:
            start: Starting value (or stop if stop is None)
            stop: Stopping value (exclusive)
            step: Step size

        Raises:
            ValueError: If step is zero
        """
        if step == 0:
            raise ValueError("step must not be zero")
        if stop is None:
            start, stop = 0, start
        self._start = start
        self._stop = stop
        self._step = step
        self._current = start

        # Precompute length for __getitem__ and __reversed__
        self._length = max(
            0,
            (self._stop - self._start + (self._step - (1 if self._step > 0 else -1)))
            // self._step,
        )

    @property
    def start(self) -> int:
        """Starting value."""
        return self._start

    @property
    def stop(self) -> int:
        """Stopping value (exclusive)."""
        return self._stop

    @property
    def step(self) -> int:
        """Step size."""
        return self._step

    def __aiter__(self) -> AsyncIterator[int]:
        """Return async iterator."""
        self._current = self._start
        return self

    async def __anext__(self) -> int:
        """Get next value asynchronously."""
        if (self._step > 0 and self._current >= self._stop) or (
            self._step < 0 and self._current <= self._stop
        ):
            raise StopAsyncIteration
        value = self._current
        self._current += self._step
        return value

    def __len__(self) -> int:
        """Return length of range."""
        return self._length

    @overload
    def __getitem__(self, key: int) -> int: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    def __getitem__(self, key: Union[int, slice]) -> Union[int, "arange"]:
        """Get item by index or slice."""
        if isinstance(key, int):
            if key < 0:
                key += self._length
            if not 0 <= key < self._length:
                raise IndexError("arange index out of range")
            return self._start + key * self._step
        elif isinstance(key, slice):
            start, stop, step = key.indices(self._length)
            new_start = self._start + start * self._step
            new_step = self._step * step
            new_stop = self._start + stop * self._step
            return arange(new_start, new_stop, new_step)
        else:
            raise TypeError("arange indices must be integers or slices")

    def __repr__(self) -> str:
        """String representation."""
        if self._step == 1:
            if self._start == 0:
                return f"arange({self._stop})"
            return f"arange({self._start}, {self._stop})"
        return f"arange({self._start}, {self._stop}, {self._step})"


async def azip(*aiters: AsyncIterator[Any]) -> AsyncIterator[tuple]:
    """Asynchronous equivalent of zip() for async iterables.

    Iterates over multiple async iterables in parallel, yielding tuples
    of items, one from each iterable. Stops as soon as the shortest iterable is exhausted.

    Args:
        *aiters: Async iterators to zip together

    Yields:
        Tuples of values from each iterator

    Example:
        >>> async for x, y in azip(gen1(), gen2()):
        ...     print(x, y)
    """
    # Create async iterators
    iterators = [ait.__aiter__() for ait in aiters]

    while True:
        try:
            # Await the next value from each iterator concurrently
            values = await asyncio.gather(*(it.__anext__() for it in iterators))
            yield tuple(values)
        except (StopAsyncIteration, asyncio.CancelledError):
            return
