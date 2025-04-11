import json
import os
from functions import Task

TASKS_FILE = "tasks.json"

def load_tasks() -> list[Task]:
    if not os.path.exists(TASKS_FILE):
        return []
    with open(TASKS_FILE, "r") as f:
        task_dicts = json.load(f)
    return [Task(**t) for t in task_dicts]

def save_task(new_task: Task):
    tasks = load_tasks()
    tasks.append(new_task)
    with open(TASKS_FILE, "w") as f:
        json.dump([t.__dict__ for t in tasks], f, indent=2)
