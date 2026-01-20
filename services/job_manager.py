"""
Job Manager Service
Handles job persistence and status tracking
"""

import os
import json
from datetime import datetime
from typing import Optional, List

class JobManager:
    def __init__(self):
        self.jobs_dir = "data/jobs"
        os.makedirs(self.jobs_dir, exist_ok=True)
        self._cache = {}

    def save_job(self, job: dict) -> None:
        """Save job to disk and cache"""
        job_id = job['id']
        self._cache[job_id] = job

        filepath = os.path.join(self.jobs_dir, f"{job_id}.json")
        with open(filepath, 'w') as f:
            json.dump(job, f, indent=2)

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job from cache or disk"""
        # Check cache first
        if job_id in self._cache:
            return self._cache[job_id]

        # Try to load from disk
        filepath = os.path.join(self.jobs_dir, f"{job_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                job = json.load(f)
                self._cache[job_id] = job
                return job

        return None

    def update_job(self, job_id: str, updates: dict) -> Optional[dict]:
        """Update job with new data"""
        job = self.get_job(job_id)
        if job:
            job.update(updates)
            self.save_job(job)
            return job
        return None

    def list_jobs(self) -> List[dict]:
        """List all jobs, sorted by creation date"""
        jobs = []

        # Load all jobs from disk
        if os.path.exists(self.jobs_dir):
            for filename in os.listdir(self.jobs_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.jobs_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            job = json.load(f)
                            jobs.append(job)
                    except:
                        pass

        # Sort by creation date, newest first
        jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return jobs

    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        # Remove from cache
        if job_id in self._cache:
            del self._cache[job_id]

        # Remove from disk
        filepath = os.path.join(self.jobs_dir, f"{job_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True

        return False
