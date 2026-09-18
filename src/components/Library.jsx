import { categories } from '../model.js';

export default function Library({ clips, allClips, current, blind, filter, onFilter, onSelect, notes, guesses }) {
  return <aside className="library">
    <div className="panelheading"><h2>Recordings</h2><span className="badge">{clips.length}</span></div>
    <label className="filterlabel" htmlFor="filter">Category</label>
    <select id="filter" value={filter} disabled={blind} onChange={e => onFilter(e.target.value)}>
      <option value="all">All sounds</option>{Object.entries(categories).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
    </select>
    <div className="clip-list" aria-label="Choose a recording">
      {clips.map((clip, i) => <button key={clip.id} className="clip" aria-pressed={current.id === clip.id} onClick={() => onSelect(clip)}>
        <span className="clip-icon">{blind ? '?' : String(Object.keys(categories).indexOf(clip.category) + 1).padStart(2, '0')}</span>
        <span><span className="clip-name">{blind ? `Mystery recording ${String(i + 1).padStart(2, '0')}` : clip.label}</span><span className="clip-sub">{blind ? 'Hidden category' : clip.id.split('-').slice(0, 3).join('-')} · {clip.duration} s</span></span>
        <span className="clip-status">{(blind ? guesses[clip.id] : notes.some(n => n.clipId === clip.id)) ? '✓' : ''}</span>
      </button>)}
    </div>
    <div className="library-footer"><span className="tiny-label">COLLECTION 01</span><p>The everyday, unheard.</p><span>ESC-50 · {Object.keys(categories).length} classes · {allClips.length} clips</span></div>
  </aside>;
}
