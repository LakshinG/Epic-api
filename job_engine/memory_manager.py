import json
import os
from typing import Dict, Any, List

class MemoryManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.answers_path = os.path.join(self.data_dir, "answers.json")
        self.pending_path = os.path.join(self.data_dir, "pending_questions.json")
        self.job_queue_path = os.path.join(self.data_dir, "job_queue.txt")

    def load_answers(self) -> Dict[str, Any]:
        """Loads the knowledge base from answers.json."""
        if not os.path.exists(self.answers_path):
            return {}
        try:
            with open(self.answers_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def log_pending_question(self, question: str, url: str):
        """Logs an unknown question to pending_questions.json."""
        pending_questions = []
        if os.path.exists(self.pending_path):
            try:
                with open(self.pending_path, "r") as f:
                    pending_questions = json.load(f)
            except json.JSONDecodeError:
                pass

        # Check if question is already logged
        for entry in pending_questions:
            if entry.get("question") == question and entry.get("url") == url:
                return

        pending_questions.append({
            "question": question,
            "url": url
        })

        with open(self.pending_path, "w") as f:
            json.dump(pending_questions, f, indent=2)

    def load_job_queue(self) -> List[str]:
        """Loads the list of URLs from job_queue.txt."""
        if not os.path.exists(self.job_queue_path):
            return []
        with open(self.job_queue_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
