import { useState } from 'react';

const companions = [
  { id: 'pikachu', name: 'Pikachu', number: '025', color: '#FDDA24' },
  { id: 'eevee', name: 'Eevee', number: '133', color: '#F1BD83' },
  { id: 'jigglypuff', name: 'Jigglypuff', number: '039', color: '#F2B6D2' },
];
const preferenceKey = 'sound-detective.companion.v1';

export default function PixelCompanion({ playing, blind, guess }) {
  const [selected, setSelected] = useState(() => {
    try { const saved = localStorage.getItem(preferenceKey); return companions.some(c => c.id === saved) ? saved : 'pikachu'; }
    catch { return 'pikachu'; }
  });
  const buddy = companions.find(c => c.id === selected);
  const mood = blind && guess ? (guess.correct ? 'celebrating' : 'thinking') : playing ? 'listening' : 'ready';
  const message = mood === 'celebrating' ? 'Great ear! You found the sound.'
    : mood === 'thinking' ? 'A new clue. Keep listening!'
    : playing ? 'Shh… there’s a story in this sound.'
    : blind ? 'A mystery awaits. Trust your ears!'
    : 'Ready when you are. Let’s listen!';
  function choose(id) {
    setSelected(id);
    try { localStorage.setItem(preferenceKey, id); } catch { /* A cosmetic preference can remain in memory. */ }
  }
  return <section className={`companion-bar ${mood}`} aria-label="Your listening companion" style={{ '--buddy-color': buddy.color }}>
    <div className="companion-scene" aria-hidden="true">
      <span className="pixel-star star-one">✦</span><span className="pixel-star star-two">✧</span>
      <img className="companion-sprite" src={`/pokemon/${buddy.id}.png`} alt="" width="96" height="96" />
    </div>
    <div className="companion-dialogue">
      <div className="companion-kicker"><span>LISTENING PARTNER</span><span className="companion-number">No. {buddy.number}</span></div>
      <p className="companion-name">{buddy.name} <span className="companion-mood">{mood === 'listening' ? '♪ listening' : mood === 'celebrating' ? '★ nice catch' : 'on your team'}</span></p>
      <p className="companion-message" aria-live="polite">{message}</p>
    </div>
    <div className="companion-party" role="group" aria-label="Choose a Pokémon partner">
      {companions.map(c => <button key={c.id} className="partner-option" aria-label={`Choose ${c.name}`} aria-pressed={selected === c.id} onClick={() => choose(c.id)} style={{ '--character-color': c.color }}>
        <img src={`/pokemon/${c.id}.png`} alt="" width="64" height="64" /><span>{c.name}</span>
      </button>)}
    </div>
  </section>;
}
