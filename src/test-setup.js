import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { JSDOM } from 'jsdom';

// Node 25 also exposes localStorage; use an isolated browser Storage instance.
vi.stubGlobal('localStorage', new JSDOM('', { url: 'http://localhost' }).window.localStorage);

const context = { setTransform() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, fillText() {}, fillRect() {}, strokeRect() {} };
vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);
vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
vi.stubGlobal('OfflineAudioContext', class {
  async decodeAudioData() { return { getChannelData: () => new Float32Array([0, .5, -.5, 0]), duration: 5, sampleRate: 44100, numberOfChannels: 1 }; }
});
vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(function () { Object.defineProperty(this, 'paused', { configurable: true, value: true }); this.dispatchEvent(new Event('pause')); });
vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(function () { Object.defineProperty(this, 'paused', { configurable: true, value: false }); this.dispatchEvent(new Event('play')); return Promise.resolve(); });
beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, arrayBuffer: async () => new ArrayBuffer(4) }));
});
afterEach(() => { cleanup(); const method = Object.getPrototypeOf(localStorage).setItem; if (vi.isMockFunction(method)) method.mockRestore(); });
