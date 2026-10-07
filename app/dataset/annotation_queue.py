import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.dataset.schemas import (
    AnnotationTask,
    AnnotationTaskStatus,
    HumanAnnotationSubmission,
    SemanticAutoSuggestions,
)
from app.training.schemas import TrainingPost


class AnnotationQueueError(ValueError):
    """Raised when an invalid task transition or queue operation is attempted."""
    pass


class AnnotationQueue:
    """
    Manages the human annotation workflow queue.
    Enforces that model-generated predictions remain suggestions (auto_suggestions)
    and only human-submitted, QC-passed records become accepted training labels.
    """

    def __init__(self):
        self.tasks: Dict[str, AnnotationTask] = {}  # task_id -> AnnotationTask
        self.post_to_task: Dict[str, str] = {}  # post_id -> task_id

    def enqueue(
        self,
        post: TrainingPost,
        auto_suggestions: Optional[SemanticAutoSuggestions] = None,
        priority: int = 1,
    ) -> AnnotationTask:
        """Adds a verified TrainingPost to the annotation queue with optional auto-suggestions."""
        if post.post_id in self.post_to_task:
            existing_task_id = self.post_to_task[post.post_id]
            return self.tasks[existing_task_id]

        task_id = f"task_{uuid.uuid4().hex[:10]}"
        task = AnnotationTask(
            annotation_task_id=task_id,
            post_id=post.post_id,
            post=post,
            status=AnnotationTaskStatus.PENDING,
            priority=priority,
            auto_suggestions=auto_suggestions,
        )

        self.tasks[task_id] = task
        self.post_to_task[post.post_id] = task_id
        return task

    def assign(self, task_id: str, annotator_id: str) -> AnnotationTask:
        """Assigns an annotator to a task and transitions status to IN_PROGRESS."""
        if task_id not in self.tasks:
            raise AnnotationQueueError(f"Task not found: {task_id}")

        task = self.tasks[task_id]
        if task.status not in (AnnotationTaskStatus.PENDING, AnnotationTaskStatus.NEEDS_REVIEW):
            raise AnnotationQueueError(f"Cannot assign task in status {task.status}")

        task.assigned_annotator = annotator_id
        task.status = AnnotationTaskStatus.IN_PROGRESS
        task.updated_at = datetime.now(timezone.utc).isoformat()
        return task

    def submit_annotation(self, task_id: str, submission: HumanAnnotationSubmission) -> AnnotationTask:
        """Submits human annotation. Transitions task to SUBMITTED."""
        if task_id not in self.tasks:
            raise AnnotationQueueError(f"Task not found: {task_id}")

        task = self.tasks[task_id]
        task.submissions.append(submission)
        task.status = AnnotationTaskStatus.SUBMITTED
        task.updated_at = datetime.now(timezone.utc).isoformat()
        return task

    def accept(self, task_id: str, consensus_submission: Optional[HumanAnnotationSubmission] = None) -> AnnotationTask:
        """Marks task as ACCEPTED with verified consensus ground truth."""
        if task_id not in self.tasks:
            raise AnnotationQueueError(f"Task not found: {task_id}")

        task = self.tasks[task_id]
        if consensus_submission:
            task.consensus_submission = consensus_submission
        elif task.submissions:
            task.consensus_submission = task.submissions[-1]
        else:
            raise AnnotationQueueError("Cannot accept a task without at least one human annotation submission.")

        task.status = AnnotationTaskStatus.ACCEPTED
        task.updated_at = datetime.now(timezone.utc).isoformat()
        return task

    def reject(self, task_id: str, reason: str) -> AnnotationTask:
        """Marks task as REJECTED. Rejected items are excluded from training datasets."""
        if task_id not in self.tasks:
            raise AnnotationQueueError(f"Task not found: {task_id}")

        task = self.tasks[task_id]
        task.status = AnnotationTaskStatus.REJECTED
        task.qc_notes.append(reason)
        task.updated_at = datetime.now(timezone.utc).isoformat()
        return task

    def flag_for_review(self, task_id: str, note: str) -> AnnotationTask:
        """Flags task for secondary review or adjudication."""
        if task_id not in self.tasks:
            raise AnnotationQueueError(f"Task not found: {task_id}")

        task = self.tasks[task_id]
        task.status = AnnotationTaskStatus.NEEDS_REVIEW
        task.qc_notes.append(note)
        task.updated_at = datetime.now(timezone.utc).isoformat()
        return task

    def get_tasks_by_status(self, status: AnnotationTaskStatus) -> List[AnnotationTask]:
        """Filters tasks by current status."""
        return [t for t in self.tasks.values() if t.status == status]

    def get_accepted_tasks(self) -> List[AnnotationTask]:
        """Returns all accepted tasks eligible for final dataset packaging."""
        return self.get_tasks_by_status(AnnotationTaskStatus.ACCEPTED)
