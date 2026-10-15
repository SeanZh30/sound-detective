export function AnnotationPanel({ selection, onSelect, duration, tag, setTag, note, setNote, editing, onSave, onReset, ready }) {
  return <aside className="notes">
    <div className="panelheading"><h2>Leave a clue</h2><span aria-hidden="true">✎</span></div>
    <p className="note-intro">Mark a sound event, or add a label to the entire recording.</p>
    <form onSubmit={e => { e.preventDefault(); onSave(); }}>
      <fieldset><legend>Time interval <span>sec</span></legend><div className="time-inputs">
        <label>Start<input id="start" type="number" min="0" max={duration} step="0.01" value={selection[0]} onChange={e => onSelect([e.target.value, selection[1]])} required /></label><span>—</span>
        <label>End<input id="end" type="number" min="0" max={duration} step="0.01" value={selection[1]} onChange={e => onSelect([selection[0], e.target.value])} required /></label>
      </div><button className="text-button" type="button" onClick={() => onSelect([0, duration])}>Select entire recording</button></fieldset>
      <label htmlFor="tag">Sound label</label><input id="tag" type="text" maxLength="40" placeholder="e.g. Short, sharp bark" value={tag} onChange={e => setTag(e.target.value)} required />
      <label htmlFor="note">Observation <span className="optional">optional</span></label><textarea id="note" maxLength="500" rows="4" placeholder="What stands out? Rhythm, intensity, background sounds…" value={note} onChange={e => setNote(e.target.value)} />
      <button className="primary" type="submit" disabled={!ready}>{editing ? 'Save changes' : '+ Save annotation'}</button>
      {editing && <button className="text-button" type="button" onClick={onReset}>Cancel editing</button>}
    </form>
    <p className="storage-note">Saved in this browser only.<br />Export a JSON backup to keep your notes.</p>
    <div className="tip"><span className="tiny-label">LISTEN CLOSER</span><h3>Listen for the little things.</h3><p>Continuous or intermittent?<br />Rhythmic or unpredictable?<br />Close by, or in the background?</p></div>
  </aside>;
}

export function AnnotationList({ notes, onJump, onEdit, onDelete }) {
  return <section className="observations">
    <div className="section-label"><span>YOUR OBSERVATIONS</span><span>{notes.length} {notes.length === 1 ? 'note' : 'notes'}</span></div><h3>Your annotations</h3>
    {notes.length === 0 ? <div className="empty-notes">No annotations yet.<br />Select a region in the waveform and leave your first clue.</div> : notes.map(n => <article key={n.id} className="annotation">
      <div className="annotation-body"><button className="annotation-time" aria-label="Jump to annotated interval" onClick={() => onJump(n)}>{n.start.toFixed(2)} — {n.end.toFixed(2)} s ↗</button><h4>{n.tag}</h4><p>{n.note}</p></div>
      <div className="annotation-actions"><button onClick={() => onEdit(n)}>Edit</button><button onClick={() => onDelete(n.id)}>Delete</button></div>
    </article>)}
  </section>;
}
