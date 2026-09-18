import { useCallback, useEffect, useRef, useState } from 'react';
import clips from './dataset.json';
import { formatTime, readAnnotations, shuffle, STORAGE_KEY, validateAnnotation } from './model.js';
import useAudio from './hooks/useAudio.js';
import Library from './components/Library.jsx';
import Waveform from './components/Waveform.jsx';
import Challenge from './components/Challenge.jsx';
import PixelCompanion from './components/PixelCompanion.jsx';
import { AnnotationList, AnnotationPanel } from './components/AnnotationPanel.jsx';

export default function App() {
  const [saved] = useState(() => readAnnotations(clips));
  const [notes, setNotes] = useState(saved.notes);
  const [message, setMessage] = useState(saved.warning);
  const notify = useCallback(text => setMessage(text), []);
  const [current, setCurrent] = useState(clips[0]);
  const [blind, setBlind] = useState(false), [blindOrder, setBlindOrder] = useState([]);
  const [filter, setFilter] = useState('all'), [guesses, setGuesses] = useState({});
  const [zoom, setZoom] = useState(1), [selection, setSelection] = useState([0, 5]);
  const [tag, setTag] = useState(''), [note, setNote] = useState(''), [editing, setEditing] = useState(null);
  const player = useAudio(current, notify);
  const latest = useRef(null);
  const order = blind ? blindOrder : clips;
  const visible = blind ? order : clips.filter(c => filter === 'all' || c.category === filter);
  const currentNotes = notes.filter(n => n.clipId === current.id).sort((a, b) => a.start - b.start);
  const next = blindOrder.find(c => !guesses[c.id] && c.id !== current.id);
  const index = order.indexOf(current) + 1;

  useEffect(() => { if (!message) return; const timer = setTimeout(() => setMessage(''), 5000); return () => clearTimeout(timer); }, [message]);
  useEffect(() => {
    const handler = e => { if (e.code === 'Space' && !e.repeat && !['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A'].includes(e.target.tagName)) { e.preventDefault(); void player.toggle(); } };
    document.addEventListener('keydown', handler); return () => document.removeEventListener('keydown', handler);
  }, [player.toggle]);
  function resetForm(duration = current.duration) { setEditing(null); setSelection([0, duration]); setTag(''); setNote(''); }
  function selectClip(clip) { setCurrent(clip); setZoom(1); resetForm(clip.duration); }
  function persist(nextNotes) {
    setNotes(nextNotes);
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(nextNotes)); return true; }
    catch { notify('Browser storage is unavailable. Notes are in memory; export a backup now.'); return false; }
  }
  function saveAnnotation(input = { start: selection[0], end: selection[1], tag, note }, editId = editing) {
    if (!player.ready) throw new Error('Please wait for the recording to finish loading.');
    const record = { ...validateAnnotation(input, current.duration), id: editId || crypto.randomUUID(), clipId: current.id, updatedAt: new Date().toISOString() };
    const updated = notes.some(n => n.id === record.id) ? notes.map(n => n.id === record.id ? record : n) : [...notes, record];
    const savedLocally = persist(updated); resetForm(); if (savedLocally) notify('Annotation saved in this browser');
    return { id: record.id, clipId: current.id, savedLocally };
  }
  function changeMode(value) {
    if (blind === value) return;
    setBlind(value); setFilter('all');
    if (value) { const nextOrder = blindOrder.length ? blindOrder : shuffle(clips); setBlindOrder(nextOrder); selectClip(nextOrder.find(c => !guesses[c.id]) || nextOrder[0]); }
    else selectClip(current);
  }
  function exportNotes() {
    if (!notes.length) { notify('Save an annotation before exporting.'); return; }
    const body = { schemaVersion: 1, dataset: 'ESC-50 demonstration subset', exportedAt: new Date().toISOString(), annotations: notes };
    const url = URL.createObjectURL(new Blob([JSON.stringify(body, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'sound-detective-annotations.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    notify('Annotations exported as JSON');
  }
  // Stable registrations delegate to the latest React state, rather than stale closures.
  useEffect(() => { latest.current = { order, current, blind, selectClip, saveAnnotation }; });
  useEffect(() => {
    const context = document.modelContext; if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const tools = [
      { name: 'list_sound_recordings', description: 'List available recordings; labels remain hidden in blind mode.', inputSchema: { type: 'object', properties: {}, additionalProperties: false }, annotations: { readOnlyHint: true }, execute: () => ({ recordings: latest.current.order.map((c, i) => ({ index: i + 1, label: latest.current.blind ? 'Hidden' : c.label, duration: c.duration })), selectedIndex: latest.current.order.indexOf(latest.current.current) + 1 }) },
      { name: 'select_sound_recording', description: 'Select a recording and begin loading its waveform. Does not play audio.', inputSchema: { type: 'object', properties: { index: { type: 'integer', minimum: 1, maximum: 12 } }, required: ['index'], additionalProperties: false }, annotations: { readOnlyHint: false }, execute: async input => { if (!Number.isInteger(input?.index) || input.index < 1 || input.index > clips.length) throw new Error('Invalid recording index'); latest.current.selectClip(latest.current.order[input.index - 1]); await new Promise(resolve => requestAnimationFrame(resolve)); return { selectedIndex: input.index, audioStatus: 'loading' }; } },
      { name: 'save_sound_annotation', description: 'Save a new time-interval annotation for the current recording in this browser.', inputSchema: { type: 'object', properties: { start: { type: 'number', minimum: 0 }, end: { type: 'number', maximum: 5 }, tag: { type: 'string', minLength: 1, maxLength: 40 }, note: { type: 'string', maxLength: 500 } }, required: ['start', 'end', 'tag'], additionalProperties: false }, annotations: { readOnlyHint: false, untrustedContentHint: true }, execute: async input => { if (!input || typeof input.start !== 'number' || typeof input.end !== 'number' || typeof input.tag !== 'string') throw new Error('Invalid annotation'); const result = latest.current.saveAnnotation(input, null); await new Promise(resolve => requestAnimationFrame(resolve)); return result; } },
    ];
    for (const tool of tools) { try { Promise.resolve(context.registerTool(tool, { signal: lifecycle.signal })).catch(() => {}); } catch { /* Optional API. */ } }
    return () => lifecycle.abort();
  }, []);

  return <>
    <header className="topbar"><a className="brand" href="./"><span className="brandmark" aria-hidden="true">▂▆▉▅</span><span>Sound Detective<small>ENVIRONMENTAL SOUND LAB</small></span></a><span className="edition">UVA DATA SCIENCE <b>HW 01</b></span><button className="outline" onClick={exportNotes}>↓ Export notes</button></header>
    <main>
      <section className="intro"><div><p className="eyebrow">LISTEN. NOTICE. DISCOVER.</p><h1>Every sound tells a story.</h1><p className="subhead">Explore real recordings. Follow the waveform. Leave your observations.</p></div><div className="mode" role="group" aria-label="Listening mode"><button aria-pressed={!blind} onClick={() => changeMode(false)}>Explore sounds</button><button aria-pressed={blind} onClick={() => changeMode(true)}>Blind challenge <span>↗</span></button></div></section>
      <PixelCompanion playing={player.playing} blind={blind} guess={guesses[current.id]} />
      <section className="workspace" aria-label="Sound exploration workspace">
        <Library clips={visible} allClips={clips} current={current} blind={blind} filter={filter} notes={notes} guesses={guesses} onSelect={selectClip} onFilter={value => { setFilter(value); if (value !== 'all' && current.category !== value) selectClip(clips.find(c => c.category === value)); }} />
        <section className="listener">
          <div className="record-heading"><div><p className="eyebrow">RECORDING {String(index).padStart(2, '0')} / {clips.length}</p><h2>{blind ? `Mystery recording ${String(index).padStart(2, '0')}` : current.label}</h2><p className="filename">{blind ? 'Listen closely. The answer is in the sound.' : `${current.id}.wav`}</p></div><span className="category-pill">{blind ? 'Category hidden' : current.category.replaceAll('_', ' ')}</span></div>
          <div className="wave-toolbar"><span><i className="signal-dot" /> Waveform</span><label htmlFor="zoom">Zoom <input id="zoom" type="range" min="1" max="5" step="1" value={zoom} onChange={e => setZoom(Number(e.target.value))} /><output>{zoom}×</output></label></div>
          <Waveform pcm={player.decoded?.pcm} duration={current.duration} position={player.position} selection={selection.map(Number)} onSelect={setSelection} onSeek={player.seek} zoom={zoom} error={player.error} />
          <div className="transport"><div className="play-group"><button className="play-button" aria-label={player.playing ? 'Pause' : 'Play'} disabled={!player.ready || !!player.error} onClick={player.toggle}>{player.playing ? 'Ⅱ' : '▶'}</button><button className="icon-button" aria-label="Restart playback" disabled={!player.ready} onClick={player.restart}>↺</button><span className="time"><b>{formatTime(player.position)}</b><span> / </span><span>{formatTime(current.duration)}</span></span></div><div className="play-options"><label htmlFor="speed" className="sr-only">Playback speed</label><select id="speed" value={player.speed} onChange={e => player.setSpeed(Number(e.target.value))}>{[.5, .75, 1, 1.5, 2].map(rate => <option value={rate} key={rate}>{rate}×</option>)}</select><button className="icon-button" aria-pressed={player.loop} aria-label="Loop recording" onClick={() => player.setLoop(!player.loop)}>↻</button><label className="volume" htmlFor="volume">Volume<input id="volume" type="range" min="0" max="1" step=".05" value={player.volume} onChange={e => player.setVolume(Number(e.target.value))} /></label></div></div>
          <label htmlFor="seek" className="sr-only">Playback position (seconds)</label><input id="seek" className="seek" type="range" min="0" max={current.duration} step=".01" value={player.position} disabled={!player.ready} onChange={e => player.seek(Number(e.target.value))} />
          <div className="metadata"><div><span>Duration</span><strong>{current.duration.toFixed(2)} <small>sec</small></strong></div><div><span>Sample rate</span><strong>44.1 <small>kHz</small></strong></div><div><span>Channels</span><strong>Mono <small>1 channel</small></strong></div><div><span>Format</span><strong>WAV <small>16-bit</small></strong></div></div>
          {blind ? <Challenge current={current} guesses={guesses} onGuess={answer => setGuesses(previous => previous[current.id] ? previous : { ...previous, [current.id]: { answer, correct: answer === current.category } })} onNext={() => next && selectClip(next)} nextAvailable={!!next} complete={Object.keys(guesses).length === clips.length} /> : <AnnotationList notes={currentNotes} onJump={n => { setSelection([n.start, n.end]); player.seek(n.start); }} onEdit={n => { setEditing(n.id); setSelection([n.start, n.end]); setTag(n.tag); setNote(n.note); }} onDelete={id => { persist(notes.filter(n => n.id !== id)); if (editing === id) resetForm(); }} />}
        </section>
        <AnnotationPanel selection={selection} onSelect={setSelection} duration={current.duration} tag={tag} setTag={setTag} note={note} setNote={setNote} editing={editing} ready={player.ready} onSave={() => { try { saveAnnotation(); } catch (error) { notify(error.message); } }} onReset={() => resetForm()} />
      </section>
      <footer><span>Real recordings from <a href="https://github.com/karolpiczak/ESC-50" target="_blank" rel="noreferrer">ESC-50 / Karol J. Piczak</a> · <a href="/ESC50-LICENSE.txt" target="_blank" rel="noreferrer">Audio credits</a> · <a href="/pokemon/CREDITS.txt" target="_blank" rel="noreferrer">Pokémon art credits</a></span><span>UVA-inspired student project · Listen → Annotate → Discover</span></footer>
    </main><audio ref={player.audioRef} preload="auto" {...player.events} />{message && <div className="toast" role="status">{message}</div>}
  </>;
}
