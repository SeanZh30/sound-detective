import { useEffect, useRef, useState } from 'react';
import testClips from '../test-clips.json';

async function readResponse(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Prediction could not be completed.');
  return data;
}

export default function PredictionPanel({ current, blind, guessed }) {
  const [sample, setSample] = useState('current');
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const request = useRef(null);
  const generation = useRef(0);
  const locked = blind && !guessed;
  useEffect(() => {
    generation.current += 1; request.current?.abort();
    setResult(null); setError(''); setBusy(false);
    return () => { generation.current += 1; request.current?.abort(); };
  }, [current.id, sample, file, blind, guessed]);
  async function predict() {
    const token = ++generation.current;
    request.current?.abort();
    const controller = new AbortController(); request.current = controller;
    setBusy(true); setError(''); setResult(null);
    try {
      let url = '/api/predict/clip', options;
      if (sample === 'upload') {
        if (!file) throw new Error('Choose a WAV file first.');
        if (file.size > 5 * 1024 * 1024) throw new Error('WAV must be smaller than 5 MB.');
        const body = new FormData(); body.append('file', file);
        url = '/api/predict/upload'; options = { body };
      } else {
        options = { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ clip_id: sample === 'current' ? current.id : sample }) };
      }
      const data = await readResponse(await fetch(url, { method: 'POST', ...options, signal: controller.signal }));
      if (token === generation.current) setResult(data);
    } catch (e) {
      if (token === generation.current && e.name !== 'AbortError') setError(e instanceof TypeError || e instanceof SyntaxError ? 'Cannot reach the prediction service. Start the backend, then try again.' : e.message);
    } finally { if (token === generation.current) setBusy(false); }
  }
  const selectedTest = testClips.find(c => c.id === sample);
  return <section className="ai-panel" aria-labelledby="prediction-title">
    <div className="ai-heading"><div><p className="eyebrow">TRAINED ON SOUND. READY TO LISTEN.</p><h2 id="prediction-title">Ask the sound detective</h2></div><span className="category-pill">6-class MLP</span></div>
    <p>A small neural network listens for dog bark, rain, sea waves, crackling fire, clock tick, or sneezing.</p>
    {locked ? <p className="ai-notice">Make your blind-challenge guess first to unlock the AI prediction.</p> : <>
      <div className="prediction-controls"><label>Predict from<select aria-label="Prediction source" value={sample} onChange={e => setSample(e.target.value)}><option value="current">Current recording</option>{testClips.map((c, i) => <option key={c.id} value={c.id}>Held-out test recording {i + 1}</option>)}<option value="upload">Upload a WAV</option></select></label>
      {sample === 'upload' && <label>WAV file<input aria-label="Upload WAV" type="file" accept=".wav,audio/wav" onChange={e => setFile(e.target.files?.[0] || null)} /></label>}
      <button className="predict-button" onClick={predict} disabled={busy || (sample === 'upload' && !file)}>{busy ? 'Listening…' : 'Identify sound'}</button></div>
      {selectedTest && <audio controls src={`/${selectedTest.file}`} aria-label="Held-out test audio" />}
      <p className="ai-caption">{sample === 'current' ? 'The original library contains training examples. Use a held-out recording to try unseen data.' : sample === 'upload' ? 'Mono/stereo WAV · 0.25–10 seconds · up to 5 MB. Uses the first 5 seconds; shorter clips are padded.' : 'Fold 5: excluded from training and model selection. Examples chosen by filename, not prediction quality.'}</p>
      {error && <p role="alert" className="ai-error">{error}</p>}
      {result && <div className="prediction-result" role="status"><p className="eyebrow">MODEL PREDICTION</p><h3>{result.label}</h3><p className="ai-caption">{selectedTest ? `Dataset label: ${selectedTest.label} · ${result.category === selectedTest.category ? 'Correct' : 'Incorrect'}` : `Source: ${result.split}`}</p><div className="probabilities">{result.probabilities.map(p => <div className="probability-row" key={p.category}><span>{p.label}</span><meter min="0" max="1" value={p.probability} aria-label={`${p.label} probability`} /><strong>{(p.probability * 100).toFixed(1)}%</strong></div>)}</div><p className="ai-caption">These are model scores, not calibrated certainty. Sounds outside the six classes will still receive one of these labels.</p></div>}
    </>}
  </section>;
}
