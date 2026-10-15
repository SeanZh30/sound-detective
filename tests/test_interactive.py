import json
import time
from uuid import uuid4
import numpy as np
import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
import backend.app as api
from backend.training_jobs import TrainingJobs, TrainingConfig
from training.features import CLASSES, FEATURE_VERSION
from training.interactive import write_status

@pytest.fixture
def manager(tmp_path, monkeypatch):
    # Hermetic learnable features: no network or full audio dataset required.
    rows = [{'filename': f'{fold}-{c}-{i}.wav', 'fold': str(fold), 'category': c,
             'src_file': f'{fold}-{c}-{i}'} for fold in range(1, 6) for c in CLASSES for i in range(2)]
    (tmp_path/'data/raw').mkdir(parents=True)
    (tmp_path/'data/cache').mkdir(parents=True)
    (tmp_path/'data/raw/manifest.json').write_text(json.dumps(rows))
    rng = np.random.default_rng(7400)
    centers = rng.normal(size=(6,168)) * 2
    X = np.stack([centers[CLASSES.index(r['category'])] + rng.normal(size=168)*0.1 for r in rows]).astype('float32')
    np.savez(tmp_path/'data/cache/features.npz', X=X, version=FEATURE_VERSION, filenames=[r['filename'] for r in rows])
    jobs = TrainingJobs(tmp_path)
    monkeypatch.setattr(api, 'jobs', jobs)
    yield jobs
    jobs.close()

def wait_done(jobs, run_id):
    deadline=time.monotonic()+30
    snapshots=[]
    while time.monotonic()<deadline:
        state=jobs.get(run_id); snapshots.append(state)
        if state['status'] in ['completed','failed','cancelled']:
            return state,snapshots
        time.sleep(.02)
    pytest.fail('Worker timed out')

def test_real_worker_progress_checkpoint_and_refresh(manager):
    client=TestClient(api.app)
    response=client.post('/api/training',json={'learning_rate':.03,'epochs':40,'hidden':32,'dropout':.2})
    assert response.status_code==202
    run_id=response.json()['id']
    assert client.post('/api/training',json={}).status_code==409
    result,snapshots=wait_done(manager,run_id)
    assert result['status']=='completed',result
    assert len(result['history'])==40
    assert result['history'][-1]['train_loss'] < result['history'][0]['train_loss']
    assert all(s['metrics'] is None for s in snapshots if s['status']!='completed')
    assert result['best_epoch']==min(result['history'],key=lambda h:h['validation_loss'])['epoch']
    assert set(result['metrics'])=={'train','validation','test'}
    assert client.get('/api/training').json()['job']['id']==run_id
    checkpoint=torch.load(manager.root/'backend/model/runs'/f'{run_id}.pt',weights_only=True)
    assert checkpoint['hidden']==32 and checkpoint['feature_version']==FEATURE_VERSION
    # A new service instance can restore a completed run after refresh/restart.
    assert TrainingJobs(manager.root).latest()['status']=='completed'

def test_stop_run_and_reject_invalid_inputs(manager):
    client=TestClient(api.app)
    for body in [{'epochs':0},{'epochs':501},{'hidden':1000},{'learning_rate':0},{'dropout':1},{'unknown':1}]:
        assert client.post('/api/training',json=body).status_code==422
    assert client.get('/api/training/not-a-uuid').status_code==422
    assert client.get(f'/api/training/{uuid4()}').status_code==404
    state=manager.start(TrainingConfig(epochs=500))
    stopped=client.post(f'/api/training/{state["id"]}/stop').json()
    assert stopped['status']=='cancelled'
    assert manager.process.poll() is not None
    assert manager.get(state['id'])['status']=='cancelled'

def test_worker_failure_is_reported(manager):
    (manager.root/'data/raw/manifest.json').write_text('invalid json')
    state=manager.start(TrainingConfig(epochs=1))
    result,_=wait_done(manager,state['id'])
    assert result['status']=='failed'
    assert 'Training failed:' in result['message']

def test_interrupted_run_is_not_left_running(manager):
    run_id=str(uuid4());folder=manager.directory/run_id;folder.mkdir(parents=True)
    write_status(folder/'status.json',{'id':run_id,'status':'running'})
    assert manager.get(run_id)['status']=='failed'

def test_previous_completed_model_survives_new_cancelled_run(manager):
    finished=str(uuid4());folder=manager.directory/finished;folder.mkdir(parents=True)
    write_status(folder/'status.json',{'id':finished,'status':'completed'})
    next_run=manager.start(TrainingConfig(epochs=500))
    manager.stop(next_run['id'])
    response=TestClient(api.app).get('/api/training').json()
    assert response['job']['status']=='cancelled'
    assert response['completed_run']==finished
