import { useEffect, useRef, useState } from 'react';

export default function Waveform({ pcm, duration, position, selection, onSelect, onSeek, zoom, error }) {
  const canvasRef = useRef(null), scrollRef = useRef(null), drag = useRef(null);
  const [width, setWidth] = useState(400);
  useEffect(() => {
    const observer = new ResizeObserver(entries => setWidth(Math.max(180, entries[0].contentRect.width)));
    observer.observe(scrollRef.current); return () => observer.disconnect();
  }, []);
  useEffect(() => { scrollRef.current.scrollLeft = 0; }, [pcm]);
  useEffect(() => {
    const canvas = canvasRef.current, ctx = canvas.getContext('2d');
    const w = width * zoom, h = 228, ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio); canvas.style.width = `${w}px`;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.strokeStyle = '#344957'; ctx.lineWidth = 1;
    for (let y = 30; y <= 190; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    const ticks = Math.max(2, Math.floor(w / 65)); ctx.font = '11px ui-monospace, monospace'; ctx.fillStyle = '#B5C5D1';
    for (let i = 0; i <= ticks; i++) {
      const x = i * w / ticks; ctx.strokeStyle = '#2C3C49'; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 200); ctx.stroke();
      ctx.textAlign = i === 0 ? 'left' : i === ticks ? 'right' : 'center'; ctx.fillText(`${(duration * i / ticks).toFixed(1)}s`, Math.max(5, Math.min(w - 5, x)), 217);
    }
    if (!pcm) return;
    const [a, b] = selection;
    if (Number.isFinite(a) && Number.isFinite(b) && b > a && a >= 0 && b <= duration && (a > 0 || b < duration)) {
      ctx.fillStyle = '#FDDA2424'; ctx.fillRect(a / duration * w, 0, (b - a) / duration * w, 200);
      ctx.strokeStyle = '#FDDA24'; ctx.strokeRect(a / duration * w, .5, (b - a) / duration * w, 199);
    }
    const step = 3, bucket = pcm.length / (w / step);
    for (let x = 0; x < w; x += step) {
      let min = 0, max = 0;
      for (let i = Math.floor(x / step * bucket); i < Math.min(pcm.length, Math.floor((x / step + 1) * bucket)); i++) { min = Math.min(min, pcm[i]); max = Math.max(max, pcm[i]); }
      ctx.fillStyle = x / w <= position / duration ? '#FDDA24' : '#E57200';
      ctx.fillRect(x, 100 - max * 86, 1.8, Math.max(1, (max - min) * 86));
    }
    ctx.fillStyle = '#EFF2F8'; ctx.fillRect(position / duration * w, 0, 1, 200);
  }, [pcm, duration, position, selection, width, zoom]);
  const eventTime = e => { const rect = canvasRef.current.getBoundingClientRect(); return Math.max(0, Math.min(duration, (e.clientX - rect.left) / rect.width * duration)); };
  const region = (a, b) => onSelect([Number(Math.min(a, b).toFixed(2)), Number(Math.max(a, b).toFixed(2))]);
  return <>
    <div ref={scrollRef} className="wave-scroll" id="wave-scroll">
      <canvas ref={canvasRef} id="waveform" aria-label="Audio waveform: click to seek, drag to select an annotation interval"
        onPointerDown={e => { if (!pcm) return; drag.current = { time: eventTime(e), x: e.clientX }; e.currentTarget.setPointerCapture(e.pointerId); }}
        onPointerMove={e => { if (drag.current && Math.abs(e.clientX - drag.current.x) > 4) region(drag.current.time, eventTime(e)); }}
        onPointerUp={e => { if (!drag.current) return; if (Math.abs(e.clientX - drag.current.x) <= 4) onSeek(eventTime(e)); else region(drag.current.time, eventTime(e)); drag.current = null; }}
        onPointerCancel={() => { drag.current = null; }} />
      {!pcm && <div id="wave-loading" className="wave-loading" role="status">{error || 'Decoding audio waveform…'}</div>}
    </div>
    <p className="wave-hint">Click to seek · Drag to select · Zoom and scroll to explore</p>
  </>;
}
