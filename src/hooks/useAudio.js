import { useCallback, useEffect, useRef, useState } from 'react';

// Own the browser media resource here; React owns every visible control.
export default function useAudio(clip, notify) {
  const audioRef = useRef(null);
  const [decoded, setDecoded] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [error, setError] = useState('');
  const [speed, setSpeed] = useState(1);
  const [volume, setVolume] = useState(0.7);
  const [loop, setLoop] = useState(false);
  const ready = decoded?.clipId === clip.id;

  useEffect(() => {
    const audio = audioRef.current;
    const abort = new AbortController();
    let active = true;
    audio.pause(); audio.currentTime = 0; audio.src = `/${clip.file}`; audio.load();
    setPlaying(false); setPosition(0); setDecoded(null); setError('');
    (async () => {
      try {
        const response = await fetch(`/${clip.file}`, { signal: abort.signal });
        if (!response.ok) throw new Error('Audio file not found.');
        const bytes = await response.arrayBuffer();
        const context = new OfflineAudioContext(1, 1, 44100);
        const buffer = await context.decodeAudioData(bytes);
        if (active) setDecoded({ clipId: clip.id, pcm: buffer.getChannelData(0), duration: buffer.duration, sampleRate: buffer.sampleRate, channels: buffer.numberOfChannels });
      } catch (err) { if (active && err.name !== 'AbortError') setError('Unable to load this recording. Select another clip or refresh to retry.'); }
    })();
    return () => { active = false; abort.abort(); audio.pause(); };
  }, [clip]);

  useEffect(() => { audioRef.current.playbackRate = speed; }, [speed, clip]);
  useEffect(() => { audioRef.current.volume = volume; }, [volume]);
  useEffect(() => { audioRef.current.loop = loop; }, [loop]);

  const seek = useCallback(seconds => {
    const value = Math.max(0, Math.min(clip.duration, seconds));
    audioRef.current.currentTime = value; setPosition(value);
  }, [clip.duration]);
  const toggle = useCallback(async () => {
    if (!ready) return;
    const audio = audioRef.current;
    try { if (audio.paused) { if (audio.ended) audio.currentTime = 0; await audio.play(); } else audio.pause(); }
    catch { notify('Playback failed. Select the recording again.'); }
  }, [ready, notify]);
  const restart = () => { seek(0); if (audioRef.current.paused) void toggle(); };
  const events = {
    onPlay: () => setPlaying(true), onPause: () => setPlaying(false), onEnded: () => setPlaying(false),
    onTimeUpdate: () => setPosition(audioRef.current.currentTime),
    onError: () => setError('Audio playback is unavailable. Select another recording or refresh.'),
  };
  return { audioRef, decoded: ready ? decoded : null, ready, playing, position, error, speed, setSpeed, volume, setVolume, loop, setLoop, seek, toggle, restart, events };
}
