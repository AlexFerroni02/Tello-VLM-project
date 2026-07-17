"""
Zero-Latency Queue Module for Robotics and IPC.
Provides the LatencyQueue class which wraps multiprocessing.Queue
and implements a drop-oldest (capped size 1) policy.
"""

import multiprocessing
from queue import Empty, Full
from typing import Any

class LatencyQueue:
    """
    A process-safe Queue wrapper optimized for real-time video processing.
    Maintains a maximum size of 1. If a new item is pushed while the queue 
    is full, the oldest item is discarded immediately to ensure zero latency.
    """
    def __init__(self):
        """
        Initializes the LatencyQueue with a maximum size of 1.
        
        Parameters:
            None
            
        Returns:
            None
        """
        # We enforce a hard limit of 1 item in the underlying multiprocessing Queue
        self.queue = multiprocessing.Queue(maxsize=1)

    def put_latest(self, item: Any):
        """
        Pushes an item into the queue. If the queue is full, it discards the 
        oldest item first and then writes the new one.
        
        Parameters:
            item (Any): The data to be queued (e.g., numpy frame).
            
        Returns:
            None
        """
        try:
            # Attempt to put the item in a non-blocking manner
            self.queue.put_nowait(item)
        except Full:
            # Queue is full, discard the old item
            try:
                self.queue.get_nowait()
            except Empty:
                # Handled race condition: another thread might have read it already
                pass
            
            # Put the new, latest item
            try:
                self.queue.put_nowait(item)
            except Full:
                # Ignored: handled concurrent write edge cases
                pass

    def get(self, block: bool = True, timeout: float = None) -> Any:
        """
        Retrieves the latest item from the queue. Blocks by default.
        
        Parameters:
            block (bool): Whether to block until an item is available.
            timeout (float): Time to block in seconds.
            
        Returns:
            Any: The retrieved item.
            
        Raises:
            Empty: If the queue is empty and the timeout expired.
        """
        return self.queue.get(block=block, timeout=timeout)

    def empty(self) -> bool:
        """
        Checks if the queue is currently empty.
        
        Parameters:
            None
            
        Returns:
            bool: True if empty, False otherwise.
        """
        return self.queue.empty()
