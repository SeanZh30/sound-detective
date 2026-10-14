"""Single-worker local job manager; status survives browser refreshes."""
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from training.interactive import write_status

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = {'queued', 'preparing', 'running', 'evaluating'}

class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    learning_rate: float = Field(default=0.03, ge=0.0001, le=0.2)
    epochs: int = Field(default=100, ge=1, le=500)
    hidden: int = Field(default=64, ge=8, le=256)
    dropout: float = Field(default=0.5, ge=0, le=0.8)

class TrainingJobs:
    def __init__(self, root=ROOT):
        self.root = root
        self.directory = root / 'training/results/runs'
        self.lock = threading.RLock()
        self.process = None
        self.active_id = None

    def get(self, run_id):
        with self.lock:
            path = self.directory / str(run_id) / 'status.json'
            if not path.exists():
                raise HTTPException(404, 'Training run not found.')
            status = json.loads(path.read_text())
            if status['status'] in ACTIVE:
                if self.active_id != str(run_id) or self.process is None or self.process.poll() is not None:
                    status.update(status='failed', message='Training worker stopped. Start a new run to retry.')
                    write_status(path, status)
            return status

    def latest(self):
        with self.lock:
            paths = list(self.directory.glob('*/status.json'))
            return self.get(max(paths, key=lambda p: p.stat().st_mtime).parent.name) if paths else None

    def latest_completed(self):
        with self.lock:
            for path in sorted(self.directory.glob('*/status.json'), key=lambda p: p.stat().st_mtime, reverse=True):
                if json.loads(path.read_text()).get('status') == 'completed':
                    return path.parent.name
            return None

    def start(self, config):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise HTTPException(409, 'A training run is already active. Wait or stop it first.')
            run_id = str(uuid4())
            folder = self.directory / run_id
            folder.mkdir(parents=True)
            status = {'id': run_id, 'status': 'queued', 'message': 'Starting training worker…',
                      'created_at': datetime.now(timezone.utc).isoformat(), 'config': config.model_dump(),
                      'epoch': 0, 'best_epoch': None, 'history': [], 'metrics': None}
            write_status(folder / 'status.json', status)
            try:
                with (folder / 'worker.log').open('w') as log:
                    self.process = subprocess.Popen([sys.executable, '-m', 'training.interactive', str(folder), '--root', str(self.root)],
                        cwd=self.root, env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}, stdout=log, stderr=subprocess.STDOUT)
                self.active_id = run_id
            except Exception:
                status.update(status='failed', message='Unable to start training worker.')
                write_status(folder / 'status.json', status)
                raise HTTPException(500, status['message'])
            return status

    def stop(self, run_id):
        with self.lock:
            status = self.get(run_id)
            if status['status'] in ACTIVE and self.active_id == str(run_id):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
                # Read the final atomic snapshot after the worker has exited.
                path = self.directory / str(run_id) / 'status.json'
                status = json.loads(path.read_text())
                if status['status'] != 'completed':
                    status.update(status='cancelled', message='Training stopped. Start a new run when ready.')
                    write_status(path, status)
            return status

    def close(self):
        with self.lock:
            if self.active_id and self.process and self.process.poll() is None:
                self.stop(self.active_id)

jobs = TrainingJobs()
