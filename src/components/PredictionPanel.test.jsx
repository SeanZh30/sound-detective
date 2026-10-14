import { act, fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import PredictionPanel from './PredictionPanel.jsx';
const current = { id: 'clip-a' };
const answer = { label: 'Rain', category: 'rain', split: 'training', probabilities: [{ category: 'rain', label: 'Rain', probability: .9 }] };
it('shows actual service probabilities and hides them when the clip changes', async () => {
  fetch.mockResolvedValue({ ok: true, json: async () => answer });
  const view = render(<PredictionPanel current={current} blind={false} guessed={false} />);
  fireEvent.click(screen.getByRole('button', { name: 'Identify sound' }));
  expect(await screen.findByRole('heading', { name: 'Rain' })).toBeInTheDocument();
  expect(screen.getByText('90.0%')).toBeInTheDocument();
  expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ clip_id: 'clip-a' });
  view.rerender(<PredictionPanel current={{ id: 'clip-b' }} blind={false} guessed={false} />);
  expect(screen.queryByRole('heading', { name: 'Rain' })).not.toBeInTheDocument();
});
it('does not reveal predictions before the blind guess', () => {
  render(<PredictionPanel current={current} blind guessed={false} />);
  expect(screen.queryByRole('button', { name: 'Identify sound' })).not.toBeInTheDocument();
  expect(fetch).not.toHaveBeenCalled();
});
it('ignores a stale prediction after selection changes', async () => {
  let finish;
  fetch.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  const view = render(<PredictionPanel current={current} blind={false} guessed={false} />);
  fireEvent.click(screen.getByRole('button', { name: 'Identify sound' }));
  view.rerender(<PredictionPanel current={{ id: 'clip-b' }} blind={false} guessed={false} />);
  await act(async () => finish({ ok: true, json: async () => answer }));
  expect(screen.queryByRole('heading', { name: 'Rain' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Identify sound' })).toBeEnabled();
});
it('shows service errors and allows a retry', async () => {
  fetch.mockRejectedValue(new TypeError('offline'));
  render(<PredictionPanel current={current} blind={false} guessed={false} />);
  fireEvent.click(screen.getByRole('button', { name: 'Identify sound' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Start the backend');
  expect(screen.getByRole('button', { name: 'Identify sound' })).toBeEnabled();
});
