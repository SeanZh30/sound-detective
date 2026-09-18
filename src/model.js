export const STORAGE_KEY = 'sound-detective.annotations.v1';
export const categories = { dog: 'Dog bark', rain: 'Rain', sea_waves: 'Sea waves', crackling_fire: 'Crackling fire', clock_tick: 'Clock tick', sneezing: 'Sneeze' };
export const formatTime = seconds => `0:${Math.max(0, seconds).toFixed(1).padStart(4, '0')}`;

export function validateAnnotation(input, duration) {
  const start = Number(input.start), end = Number(input.end);
  const tag = String(input.tag || '').trim(), note = String(input.note || '').trim();
  if (input.start === '' || input.end === '' || !Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end > duration || end <= start) {
    throw new Error(`Choose an interval with 0 ≤ start < end ≤ ${duration.toFixed(2)} seconds.`);
  }
  if (!tag || tag.length > 40 || note.length > 500) throw new Error('Use a label of 1–40 characters and a note of up to 500 characters.');
  return { start, end, tag, note };
}

export function readAnnotations(clips) {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    if (!Array.isArray(saved)) throw new Error();
    const notes = saved.filter(item => {
      const clip = clips.find(c => c.id === item?.clipId);
      if (!clip || typeof item.id !== 'string' || typeof item.tag !== 'string' || typeof item.note !== 'string') return false;
      try { validateAnnotation(item, clip.duration); return true; } catch { return false; }
    });
    return { notes, warning: notes.length === saved.length ? '' : 'Some invalid saved annotations were skipped. Export your current notes as a backup.' };
  } catch { return { notes: [], warning: 'Saved annotations could not be read. Export new notes to keep a backup.' }; }
}

export function shuffle(items) {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [result[i], result[j]] = [result[j], result[i]]; }
  return result;
}
