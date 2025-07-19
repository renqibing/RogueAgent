import asyncio
import json
from collections import deque

import asyncio
from collections import deque

import asyncio

class TaskBlackboard:
    def __init__(self, max_tasks=5):
        """Initialize the TaskBlackboard with a maximum number of tasks."""
        self.max_tasks = max_tasks
        self.tasks = {}  # Using dict to store tasks (task_id -> task details)
        self.lock = asyncio.Lock()  # Lock for thread-safe operations

    async def read(self):
        """Read the tasks in the communication channel."""
        async with self.lock:
            return list(self.tasks.values())  # Read tasks under lock

    async def write(self, task):
        """Write a task to the communication channel."""
        async with self.lock:
            if len(self.tasks) >= self.max_tasks:
                # If the number of tasks exceeds the maximum, remove the oldest task
                task_id_to_remove = min(self.tasks.keys())  # Get the task with the smallest ID (oldest)
                del self.tasks[task_id_to_remove]  # Remove the oldest task
            self.tasks[task["task_id"]] = task  # Add the new task by task_id

    async def del_task(self, task_id):
        """delete the task."""
        async with self.lock:
            del self.tasks[task_id]
            return True