import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import App from './App.jsx';
import clips from './dataset.json';
import { STORAGE_KEY } from './model.js';

async function mount() {
  const view = render(<App />);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Play', exact: true })).toBeEnabled());
  return view;
}
const fill = (label, value) => fireEvent.change(screen.getByLabelText(label), { target: { value } });
async function annotate() {
  fill('Start', '0.5'); fill('End', '2'); fill('Sound label', 'Short bark');
  fireEvent.click(screen.getByRole('button', { name: '+ Save annotation' }));
  await screen.findByRole('heading', { name: 'Short bark' });
}

describe('React audio viewer', () => {
  it('loads real metadata, filters clips, and keeps playback speed when selecting a new clip', async () => {
    await mount();
    expect(screen.getByRole('heading', { name: 'Dog bark', exact: true })).toBeInTheDocument();
    fill('Playback speed', '1.5'); fill('Category', 'rain');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Play', exact: true })).toBeEnabled());
    expect(screen.getByRole('heading', { name: 'Rain', exact: true })).toBeInTheDocument();
    expect(screen.getByLabelText('Playback speed')).toHaveValue('1.5');
    expect(document.querySelector('audio').playbackRate).toBe(1.5);
    expect(screen.getAllByRole('button', { pressed: false }).filter(b => b.className === 'clip')).toHaveLength(1);
  });
  it('plays, pauses, seeks, and toggles looping through React controls', async () => {
    await mount();fireEvent.click(screen.getByRole('button', { name: 'Play', exact: true }));
    await screen.findByRole('button', { name: 'Pause', exact: true });
    fireEvent.click(screen.getByRole('button', { name: 'Pause', exact: true }));
    expect(screen.getByRole('button', { name: 'Play', exact: true })).toBeEnabled();
    fill('Playback position (seconds)', '2.5');
    expect(document.querySelector('audio').currentTime).toBe(2.5);
    fireEvent.click(screen.getByRole('button', { name: 'Loop recording' }));
    expect(document.querySelector('audio').loop).toBe(true);
  });
  it('preserves annotations saved by the original vanilla application', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([{ id: 'legacy', clipId: clips[0].id, start: 1, end: 2, tag: 'Legacy annotation', note: 'Keep me' }]));
    await mount();expect(screen.getByRole('heading', { name: 'Legacy annotation' })).toBeInTheDocument();
    expect(screen.getByText('Keep me')).toBeInTheDocument();
  });
  it('saves, edits, reloads, and deletes an interval annotation without losing other recordings', async () => {
    let view = await mount(); await annotate();
    fireEvent.click(screen.getByRole('button', { name: 'Edit', exact: true }));fill('Sound label', 'Edited bark');
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY))).toMatchObject([{ start: .5, end: 2, tag: 'Edited bark' }]);
    fill('Category', 'rain');expect(screen.queryByRole('heading', { name: 'Edited bark' })).not.toBeInTheDocument();
    view.unmount();view = await mount();expect(screen.getByRole('heading', { name: 'Edited bark' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Delete', exact: true }));
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY))).toEqual([]);view.unmount();
  });
  it('rejects backwards intervals and retains the draft', async () => {
    await mount(); fill('Start', '4');fill('End', '1');fill('Sound label', 'Invalid');
    fireEvent.click(screen.getByRole('button', { name: '+ Save annotation' }));
    expect(screen.getByText(/Choose an interval with/)).toHaveAttribute('role', 'status');
    expect(screen.getByLabelText('Sound label')).toHaveValue('Invalid');
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });
  it('reports unavailable storage while keeping the new annotation in memory', async () => {
    await mount();vi.spyOn(Object.getPrototypeOf(localStorage), 'setItem').mockImplementation(() => { throw new Error('Quota exceeded'); });
    await annotate();expect(screen.getByText(/Browser storage is unavailable/)).toHaveAttribute('role', 'status');
    expect(screen.getByRole('heading', { name: 'Short bark' })).toBeInTheDocument();
  });
  it('hides metadata and annotations in blind mode and counts each first guess only once', async () => {
    await mount();await annotate();
    fireEvent.click(screen.getByRole('button', { name: /Blind challenge/ }));
    expect(screen.getByLabelText('Category')).toBeDisabled();
    expect(screen.queryByText(`${clips[0].id}.wav`)).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Short bark' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Dog bark', exact: true }));
    expect(screen.getByText(/Correct: [01] \/ 1/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Dog bark', exact: true })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: /Next recording/ }));
    expect(screen.getByRole('button', { name: 'Dog bark', exact: true })).toBeEnabled();
    expect(screen.getByText(/Correct: [01] \/ 1/)).toBeInTheDocument();
  });
  it('shows an actionable audio error without enabling playback', async () => {
    fetch.mockResolvedValue({ ok: false });render(<App />);
    await screen.findByText(/Unable to load this recording/);
    expect(screen.getByRole('button', { name: 'Play', exact: true })).toBeDisabled();
  });
  it('exports stored intervals and labels as a versioned JSON backup', async () => {
    await mount();await annotate();
    let exported;
    const original = URL.createObjectURL;
    URL.createObjectURL = vi.fn(blob => { exported = blob; return 'blob:test-backup'; });
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function () {
      expect(this.download).toBe('sound-detective-annotations.json');
    });
    try {
      fireEvent.click(screen.getByRole('button', { name: /Export notes/ }));
      const text = await exported.text();
      expect(JSON.parse(text)).toMatchObject({ schemaVersion: 1, annotations: [{ clipId: clips[0].id, start: .5, end: 2, tag: 'Short bark' }] });
    } finally { click.mockRestore(); URL.createObjectURL = original; }
  });
  it('ignores a stale decode after the user switches recordings', async () => {
    let finishOld;let calls = 0;
    fetch.mockImplementation(() => ++calls === 1 ? new Promise(resolve => { finishOld = resolve; }) : Promise.resolve({ ok: true, arrayBuffer: async () => new ArrayBuffer(4) }));
    render(<App />);fill('Category', 'rain');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Play', exact: true })).toBeEnabled());
    await act(async () => finishOld({ ok: false }));
    expect(screen.getByRole('heading', { name: 'Rain', exact: true })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Play', exact: true })).toBeEnabled();
  });
});
