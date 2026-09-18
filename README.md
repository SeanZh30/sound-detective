# Sound Detective

A React web application for **DS7400 Deep Learning — Homework 1**. Explore real environmental recordings, inspect waveforms, annotate sound events, and try a blind-listening challenge.

## Run locally

Install Node.js 22.12 or later (Node 24 LTS recommended) and npm, then:

```sh
git clone https://github.com/SeanZh30/sound-detective.git
cd sound-detective
npm ci
npm run dev
```

Open http://127.0.0.1:5173. Stop with Ctrl+C. All 12 audio samples are bundled; after installation, the app works without third-party network requests.

```sh
npm test          # React interaction and regression tests
npm run build     # Production files in dist/
npm run preview   # Preview the build at http://127.0.0.1:4173
```

## What you can do

- Choose Pikachu, Eevee, or Jigglypuff as a pixel-art listening companion. Your choice stays in this browser. Playback triggers a gentle bounce; blind-challenge feedback changes the companion dialogue without revealing an unanswered clip. Animations respect reduced-motion preferences.
- Browse 12 recordings, filter by category, and inspect the filename and audio metadata.
- Play/pause, restart, seek, adjust volume and speed, and loop audio.
- Inspect a real waveform decoded from WAV samples. Zoom from 1× to 5×, scroll horizontally, click to seek, or drag to select a time interval.
- Use keyboard-accessible seek and interval inputs as alternatives to pointer dragging. Space toggles playback outside form controls.
- Add labels and notes to an interval or the entire clip. Edit, delete, or jump to existing annotations.
- Save annotations in browser localStorage and export them as JSON. Original annotations from version 1 are preserved using the same storage key and schema.
- Switch to a randomized blind-listening challenge. Categories, filenames, and existing annotations are hidden. Only the first guess per recording counts during the page session.

Challenge answers come from dataset labels, not a model. This is a learning activity rather than a secure assessment; the source assets contain labels.

## Data and attribution

The app includes **12 unmodified recordings from ESC-50**, two per category: dog bark, rain, sea waves, crackling fire, clock tick, and sneezing. Each is 5 seconds, mono, 44.1 kHz, 16-bit PCM WAV.

- Dataset: [ESC-50 by Karol J. Piczak](https://github.com/karolpiczak/ESC-50).
- Citation: K. J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia, 2015. [DOI](https://doi.org/10.1145/2733373.2806390).
- Dataset license: CC BY-NC; ESC-10 subset clips have CC BY terms. Preserve the [upstream license and individual recording attributions](public/ESC50-LICENSE.txt) when sharing this project. The WAVs are unchanged.
- Metadata in `src/dataset.json` retains original filenames, source recording IDs, takes, categories, and fold IDs.
- These are fold-1 demonstration samples, **not a training/test split**. Two sea-wave clips share a source recording. For later model training, use the full dataset and official folds; avoid source-recording leakage.

Restore any missing WAVs with `python3 scripts/download-samples.py` (requires Python 3 and internet).

## React architecture

| File | Responsibility |
| --- | --- |
| `src/App.jsx` | React application state, mode switching, annotation persistence/export, optional WebMCP integration |
| `src/components/Library.jsx` | Category filter and recording selection |
| `src/components/Waveform.jsx` | Canvas drawing, resizing, zoom, seeking, interval selection |
| `src/components/AnnotationPanel.jsx` | Controlled annotation form and annotation list |
| `src/components/PixelCompanion.jsx` | Partner selection, local preference, playback/answer feedback |
| `src/components/Challenge.jsx` | Answer choices, feedback, and score |
| `src/hooks/useAudio.js` | Audio element, playback settings, asynchronous decoding, cancellation/cleanup |
| `src/model.js` | Annotation validation, reading legacy saved notes, shared helpers |
| `src/styles.css` | Responsive layout and UVA-inspired palette |
| `public/audio/` | Bundled original sample recordings |
| `src/App.test.jsx` | Interaction and regression tests |

**Stack:** React, React DOM, Vite, JavaScript, CSS, HTMLAudioElement, Web Audio API, Canvas 2D, and localStorage. Tests use Vitest, Testing Library, and jsdom. Exact dependency versions are pinned in `package.json` and `package-lock.json`.

React owns the UI and interaction state through components and hooks. Canvas is the only imperative drawing surface; audio is managed through a ref with effect cleanup. Aborted/stale requests cannot replace the waveform after switching clips.

The optional `document.modelContext` integration exposes listing recordings, selecting one, and saving an annotation to compatible agents. Unsupported browsers retain all normal UI features.

## UVA / School of Data Science visual direction

The palette is inspired by the [UVA School of Data Science brand resources](https://datascience.virginia.edu/pages/school-brand-resources) and its June 2026 brand guide:

| Color | Hex | Application |
| --- | --- | --- |
| SDS Navy | `#24323E` | Main listening workspace |
| SDS Deep Navy | `#10181F` | Page background and dark surfaces |
| UVA Orange | `#E57200` | Playback, primary actions, waveform |
| SDS Yellow | `#FDDA24` | Selected interval, progress, small highlights |
| SDS Gray | `#EFF2F8` | Main text |

This is a student project with UVA-inspired colors, not an official university product. No institutional logos are reproduced. Franklin Gothic is used if installed, with system sans-serif fallbacks; no proprietary font files are bundled.

## HW1 scope and limitations

- **Interactive data viewer:** implemented.
- **Interactive annotation:** implemented, including time intervals and JSON export.
- **Backend/database:** not implemented. localStorage is browser storage and does not qualify as the backend/database extra-credit item.
- **Deep learning:** not implemented or required for HW1. Future work can add a sound classifier and later FastAPI/SQLite persistence.

Annotations remain on the current browser and origin. Localhost and the hosted app have separate storage. Export notes before clearing browser data or moving to another browser. Audio decoding and playback are tested with browser API mocks in automated tests; tests do not evaluate actual speaker output.

## Pixel-art credits

Pikachu, Eevee, and Jigglypuff are displayed using unmodified Generation V sprites from [PokeAPI/sprites](https://github.com/PokeAPI/sprites). The original PNG files, upstream repository license, and per-file source URLs are bundled in [`public/pokemon/`](public/pokemon/CREDITS.txt). Pokémon artwork and characters remain the property of their respective owners. This is an unaffiliated non-commercial student project. Companion images are decorative and do not classify or alter the audio data.
