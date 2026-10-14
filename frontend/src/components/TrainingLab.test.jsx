import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import TrainingLab from './TrainingLab.jsx';
const config = { learning_rate: .03, epochs: 100, hidden: 64, dropout: .5 };
const job = { id: 'run-12345', status: 'running', message: 'Epoch 1/100', config, epoch: 1, best_epoch: 1, metrics: null, history: [{ epoch: 1, train_loss: 1.5, validation_loss: 1.6, train_accuracy: .3, validation_accuracy: .2 }] };
it('submits changed settings and shows live metrics without exposing test scores', async () => {
  fetch.mockImplementation((url, options) => Promise.resolve({ ok: true, json: async () => options?.method === 'POST' ? job : { job: null } }));
  render(<TrainingLab onModelReady={vi.fn()} />);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Start training' })).toBeEnabled());
  fireEvent.change(screen.getByLabelText('Learning rate'), { target: { value: '.01' } });
  fireEvent.change(screen.getByLabelText('Hidden units'), { target: { value: '128' } });
  fireEvent.click(screen.getByRole('button', { name: 'Start training' }));
  await screen.findByText('RUNNING');
  const post = fetch.mock.calls.find(([, options]) => options?.method === 'POST');
  expect(JSON.parse(post[1].body)).toMatchObject({ learning_rate: .01, hidden: 128 });
  expect(screen.getByLabelText('Training progress')).toHaveAttribute('value', '1');
  expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
  expect(screen.getAllByText('Pending')).toHaveLength(2);
  expect(screen.getByRole('img', { name: 'Training and validation loss curve' })).toBeInTheDocument();
});
it('restores completed metrics and makes the new model available', async () => {
  const ready = vi.fn();
  const metrics = Object.fromEntries(['train', 'validation', 'test'].map(n => [n, { loss: .4, accuracy: .8, macro_f1: .79 }]));
  fetch.mockResolvedValue({ ok: true, json: async () => ({ job: { ...job, status: 'completed', metrics } }) });
  render(<TrainingLab onModelReady={ready} />);
  await screen.findByText('COMPLETED');
  expect(ready).toHaveBeenCalledWith(job.id);
  expect(screen.getAllByText('80.0%')).toHaveLength(3);
  expect(screen.getByText(/Your trained model is ready/)).toBeInTheDocument();
});
it('stops the active worker from the UI', async () => {
  fetch.mockImplementation((url, options) => Promise.resolve({ ok: true, json: async () => options?.method === 'POST' ? { ...job, status: 'cancelled' } : { job } }));
  render(<TrainingLab onModelReady={vi.fn()} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Stop training' }));
  await screen.findByText('CANCELLED');
  expect(fetch.mock.calls.some(([url, options]) => url === `/api/training/${job.id}/stop` && options.method === 'POST')).toBe(true);
});
