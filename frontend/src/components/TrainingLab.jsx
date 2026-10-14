import { useEffect, useRef, useState } from 'react';
import { Curve } from './TrainingReport.jsx';
const activeStates = ['queued', 'preparing', 'running', 'evaluating'];
const percent = value => `${(100 * value).toFixed(1)}%`;
async function read(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Invalid training settings. Check the input ranges.');
  return data;
}
export default function TrainingLab({ onModelReady }) {
  const [config, setConfig] = useState({ learning_rate: 0.03, epochs: 100, hidden: 64, dropout: 0.5 });
  const [job, setJob] = useState(null), [error, setError] = useState('');
  const [connectionError, setConnectionError] = useState('');
  const [connected, setConnected] = useState(false), [working, setWorking] = useState(false);
  const generation = useRef(0), action = useRef(false);
  useEffect(() => {
    const controller = new AbortController(); let timer;
    async function poll() {
      const token = generation.current;
      try {
        const data = await read(await fetch('/api/training', { signal: controller.signal }));
        if (!controller.signal.aborted && token === generation.current && !action.current) {
          setJob(data.job); setConnected(true); setConnectionError('');
          if (data.completed_run) onModelReady(data.completed_run);
        }
      } catch (e) {
        if (!controller.signal.aborted && token === generation.current) {
          setConnected(false); setConnectionError('Training service is unavailable. Start the backend; this panel reconnects automatically.');
        }
      } finally { if (!controller.signal.aborted) timer = setTimeout(poll, 350); }
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [onModelReady]);
  useEffect(() => { if (job?.status === 'completed') onModelReady(job.id); }, [job?.id, job?.status, onModelReady]);
  const active = job && activeStates.includes(job.status);
  async function submit(event) {
    event.preventDefault();
    action.current = true; generation.current += 1; setWorking(true); setError('');
    try { setJob(await read(await fetch('/api/training', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config) }))); }
    catch (e) { setError(e.message); }
    finally { action.current = false; generation.current += 1; setWorking(false); }
  }
  async function stop() {
    action.current = true; generation.current += 1; setWorking(true); setError('');
    try { setJob(await read(await fetch(`/api/training/${job.id}/stop`, { method: 'POST' }))); }
    catch (e) { setError(e.message); }
    finally { action.current = false; generation.current += 1; setWorking(false); }
  }
  const latest = job?.history.at(-1);
  const metrics = job?.metrics || (latest ? { train: { loss: latest.train_loss, accuracy: latest.train_accuracy }, validation: { loss: latest.validation_loss, accuracy: latest.validation_accuracy } } : {});
  return <section className="ai-panel" aria-labelledby="training-lab-title">
    <div className="ai-heading"><div><p className="eyebrow">CHANGE A SETTING. WATCH IT LEARN.</p><h2 id="training-lab-title">Interactive training lab</h2></div><span className="category-pill">Live MLP training</span></div>
    <p>Train a new MLP with your own settings. The monitor updates while the model learns; the original MLP and PANNs models remain available.</p>
    <form onSubmit={submit} className="training-form">
      <fieldset disabled={!!active || working || !connected} className="prediction-controls">
        <legend className="sr-only">MLP training settings</legend>
        <label>Learning rate<input aria-label="Learning rate" type="number" min="0.0001" max="0.2" step="0.0001" required value={config.learning_rate} onChange={e => setConfig({ ...config, learning_rate: e.target.value === '' ? '' : Number(e.target.value) })} /></label>
        <label>Epochs<input aria-label="Epochs" type="number" min="1" max="500" step="1" required value={config.epochs} onChange={e => setConfig({ ...config, epochs: e.target.value === '' ? '' : Number(e.target.value) })} /></label>
        <label>Hidden units<input aria-label="Hidden units" type="number" min="8" max="256" step="1" required value={config.hidden} onChange={e => setConfig({ ...config, hidden: e.target.value === '' ? '' : Number(e.target.value) })} /></label>
        <label>Dropout<input aria-label="Dropout" type="number" min="0" max="0.8" step="0.05" required value={config.dropout} onChange={e => setConfig({ ...config, dropout: e.target.value === '' ? '' : Number(e.target.value) })} /></label>
        <button className="predict-button" type="submit">Start training</button>
      </fieldset>
    </form>
    {active && <button className="outline" disabled={working} onClick={stop}>Stop training</button>}
    {(error || connectionError) && <p role="alert" className="ai-error">{error || connectionError}</p>}
    <p className="ai-caption">Uses 144 training and 48 validation clips. First-time setup downloads public training audio if needed. Test metrics appear only after the run finishes; use validation results to compare settings.</p>
    {job ? <>
      <p className="training-status" role="status"><strong>{job.status.toUpperCase()}</strong> · {job.message}</p>
      <progress aria-label="Training progress" value={job.epoch} max={job.config.epochs} /><p className="ai-caption">Epoch {job.epoch} / {job.config.epochs} · Best validation epoch: {job.best_epoch ?? '—'} · Run {job.id.slice(0, 8)}</p>
      <p className="ai-caption">Run settings: LR {job.config.learning_rate} · {job.config.hidden} hidden units · dropout {job.config.dropout} · SGD with momentum · seed 7400.</p>
      {!!job.history.length && <><div className="curve-grid"><Curve history={job.history} metric="loss" /><Curve history={job.history} metric="accuracy" /></div><p className="curve-legend"><span>● Training</span><span>● Validation</span></p></>}
      <div className="table-scroll"><table><caption>{job.status === 'completed' ? 'Final metrics — best validation checkpoint' : 'Live metrics — latest completed epoch'}</caption><thead><tr><th>Split</th><th>Loss</th><th>Accuracy</th><th>Macro F1</th></tr></thead><tbody>{['train', 'validation', 'test'].map(name => <tr key={name}><th>{name}</th><td>{metrics[name]?.loss.toFixed(3) ?? 'Pending'}</td><td>{metrics[name] ? percent(metrics[name].accuracy) : 'Pending'}</td><td>{metrics[name]?.macro_f1?.toFixed(3) ?? 'After completion'}</td></tr>)}</tbody></table></div>
      {job.status === 'completed' && <p className="ai-notice">Your trained model is ready. In “Ask the sound detective”, choose “Interactive MLP · latest completed run” to predict a recording or upload a WAV.</p>}
    </> : <p className="ai-notice">Set your parameters and start a run to see live loss and accuracy curves.</p>}
  </section>;
}
