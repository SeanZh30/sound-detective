import { categories } from '../model.js';

export default function Challenge({ current, guesses, onGuess, onNext, nextAvailable, complete }) {
  const guess = guesses[current.id], results = Object.values(guesses);
  return <section className="challenge">
    <div className="section-label"><span>LISTEN & GUESS</span><span>Correct: {results.filter(g => g.correct).length} / {results.length}</span></div>
    <h3>What do you hear?</h3><p>Listen first, then choose a category. Only your first guess counts for each recording.</p>
    <div className="guess-options">{Object.entries(categories).map(([key, label]) => <button key={key} disabled={!!guess} className={guess && key === current.category ? 'correct' : guess && !guess.correct && key === guess.answer ? 'wrong' : ''} onClick={() => onGuess(key)}>{label}</button>)}</div>
    <p className="guess-result" role="status">{guess ? `${guess.correct ? 'Correct!' : 'Not quite.'} Dataset label: ${current.label}.` : 'Choose a category to reveal the answer.'}</p>
    <button className="outline" disabled={!nextAvailable} onClick={onNext}>{complete ? 'Challenge complete' : 'Next recording →'}</button>
  </section>;
}
