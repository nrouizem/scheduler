from models import TaskDB, SessionLocal
from functions import Task

def load_tasks() -> list[Task]:
    db = SessionLocal()
    tasks = db.query(TaskDB).all()
    db.close()
    return [Task(
        name=t.name,
        estimated_duration=t.estimated_duration,
        required_focus=t.required_focus,
        category=t.category,
        flexibility=t.flexibility
    ) for t in tasks]

def save_task(new_task: Task):
    db = SessionLocal()
    db_task = TaskDB(
        name=new_task.name,
        estimated_duration=new_task.estimated_duration,
        required_focus=new_task.required_focus,
        category=new_task.category,
        flexibility=new_task.flexibility
    )
    db.add(db_task)
    db.commit()
    db.close()
