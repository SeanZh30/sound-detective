import { useEffect, useState } from 'react';
const percent = value => `${(value * 100).toFixed(1)}%`;

export function Curve({ history, metric }) {
  const width = 600, height = 190, pad = 30;
  const max = metric === 'accuracy' ? 1 : Math.max(0.001, ...history.flatMap(h => [h.train_loss, h.validation_loss])) * 1.05;
  const points = name => history.map((h, i) => `${pad + i / Math.max(1, history.length - 1) * (width - 2 * pad)},${height - pad - h[`${name}_${metric}`] / max * (height - 2 * pad)}`).join(' ');
  return <figure><figcaption>{metric === 'loss' ? 'Cross-entropy loss' : 'Accuracy'} vs. epoch</figcaption><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Training and validation ${metric} curve`}><path d={`M${pad},15 V${height - pad} H${width - pad}`} fill="none" stroke="#91a1af" />{['train', 'validation'].map((name, i) => <polyline key={name} points={points(name)} fill="none" stroke={i ? '#FDDA24' : '#E57200'} strokeWidth="2.5" />)}<text x="0" y="20">{max.toFixed(1)}</text><text x="12" y={height - pad}>0</text><text x={pad} y={height - 6}>1</text><text x={width - 55} y={height - 6}>{history.length}</text><text x={width / 2} y={height - 6}>Epoch</text></svg></figure>;
}
export default function TrainingReport() {
  const [report, setReport] = useState(null), [error, setError] = useState(''), [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setError('');
    fetch('/api/metrics', { signal: controller.signal }).then(async response => { if (!response.ok) throw new Error(); return response.json(); }).then(data => { if (!controller.signal.aborted) setReport(data); }).catch(() => { if (!controller.signal.aborted) setError('Start the backend to load the saved training report.'); });
    return () => controller.abort();
  }, [attempt]);
  return <section className="ai-panel" aria-labelledby="report-title"><p className="eyebrow">THE EVIDENCE BEHIND THE ANSWER</p><h2 id="report-title">Training & evaluation</h2>{error ? <p>{error} <button onClick={() => setAttempt(x => x + 1)}>Retry report</button></p> : !report ? <p>Loading experiment results…</p> : <>
    <p>{report.architecture} · SGD with momentum · selected epoch {report.config.epoch}</p>
    <div className="metric-cards"><div><span>Held-out MLP accuracy</span><strong>{percent(report.mlp.test.accuracy)}</strong><small>{report.counts.test} test recordings</small></div><div><span>Logistic regression</span><strong>{percent(report.baseline.test.accuracy)}</strong><small>Same features and split</small></div><div><span>{report.pretrained ? 'Fine-tuned PANNs accuracy' : 'MLP macro F1'}</span><strong>{report.pretrained ? percent(report.pretrained.metrics.test.accuracy) : report.mlp.test.macro_f1.toFixed(3)}</strong><small>{report.pretrained ? 'AudioSet pretraining · same test split' : 'Equal weight per class'}</small></div></div>
    <p className="ai-caption">240 recordings · Train: folds 1–3 ({report.counts.train}) · Validation: fold 4 ({report.counts.validation}) · Test: fold 5 ({report.counts.test}). Model selection uses validation data only.</p>
    <p>{report.beats_baseline ? 'The MLP has a lower test error rate than the shallow baseline.' : 'The MLP does not outperform the shallow baseline on this test split.'}</p>
    <h3>MLP trained from scratch</h3>
    <div className="curve-grid"><Curve history={report.history} metric="loss" /><Curve history={report.history} metric="accuracy" /></div><p className="curve-legend"><span>● Training</span><span>● Validation</span></p>
    <div className="table-scroll"><table><caption>Selected checkpoint performance</caption><thead><tr><th>Split</th><th>MLP accuracy</th><th>MLP loss</th><th>Baseline accuracy</th></tr></thead><tbody>{['train', 'validation', 'test'].map(name => <tr key={name}><th>{name}</th><td>{percent(report.mlp[name].accuracy)}</td><td>{report.mlp[name].loss.toFixed(3)}</td><td>{percent(report.baseline[name].accuracy)}</td></tr>)}</tbody></table></div>
    <details><summary>Inspect test confusion matrix and experiment protocol</summary><p>Rows are actual labels; columns are model predictions. Each class has eight test recordings.</p><div className="table-scroll"><table><thead><tr><th>Actual / predicted</th>{report.labels.map(l => <th key={l}>{l}</th>)}</tr></thead><tbody>{report.mlp.test.confusion_matrix.map((row, i) => <tr key={i}><th>{report.labels[i]}</th>{row.map((n, j) => <td key={j} className={i === j ? 'diagonal' : ''}>{n}</td>)}</tr>)}</tbody></table></div><p>{report.selection} Seed: {report.seed}. Features: 64 log-mel and 20 MFCC means/standard deviations; scaling is fitted on training data only.</p><p>Curves show the selected run, including epochs after the best checkpoint. This is a saved experiment report, not a live training monitor. A single 48-recording test fold gives a limited estimate of generalization.</p></details>
    {report.pretrained && <>
      <h3>PANNs Cnn6 · partial fine-tuning</h3>
      <p>AudioSet-pretrained convolutional network. Blocks 1–3 are frozen; block 4, the pretrained dense layer, and the new six-class head are trained. All batch-normalization running statistics stay fixed.</p>
      <p className="ai-caption">AdamW · selected epoch {report.pretrained.config.epoch} · {report.pretrained.trainable_parameters.toLocaleString()} updated parameters. Both models use the same split, but PANNs has additional pretraining data and a different architecture.</p>
      <div className="curve-grid"><Curve history={report.pretrained.history} metric="loss" /><Curve history={report.pretrained.history} metric="accuracy" /></div><p className="curve-legend"><span>● Training</span><span>● Validation</span></p>
      <div className="table-scroll"><table><caption>Fine-tuned checkpoint performance</caption><thead><tr><th>Split</th><th>Accuracy</th><th>Loss</th><th>Macro F1</th></tr></thead><tbody>{['train', 'validation', 'test'].map(name => <tr key={name}><th>{name}</th><td>{percent(report.pretrained.metrics[name].accuracy)}</td><td>{report.pretrained.metrics[name].loss.toFixed(3)}</td><td>{report.pretrained.metrics[name].macro_f1.toFixed(3)}</td></tr>)}</tbody></table></div>
      <details><summary>Fine-tuned PANNs test confusion matrix</summary><p>Rows: actual labels. Columns: predictions.</p><div className="table-scroll"><table><thead><tr><th>Actual / predicted</th>{report.labels.map(l => <th key={l}>{l}</th>)}</tr></thead><tbody>{report.pretrained.metrics.test.confusion_matrix.map((row, i) => <tr key={i}><th>{report.labels[i]}</th>{row.map((n, j) => <td key={j} className={i === j ? 'diagonal' : ''}>{n}</td>)}</tr>)}</tbody></table></div></details>
    </>}
  </>}</section>;
}
