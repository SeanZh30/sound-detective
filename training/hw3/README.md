# HW3: Does temporal information need a learned gate?

This experiment extends the HW2 environmental-sound MLP to **UrbanSound8K (8,732 clips, 10 classes)**, then tests global/temporal feature fusion. HW2's six-class app, weights, and results remain unchanged. The HW3 models predict a different label set and are not loaded into the HW2 web app.

## Hypothesis, fixed before results

The HW2 representation averages over time, discarding temporal structure. Four ordered segments may preserve short events and changes within a recording. An input-dependent gate may suppress unreliable segment features while retaining useful local evidence. Expected failure modes include overfitting, sigmoid saturation, sensitivity to segment boundaries, and padded short clips. Gates are routing coefficients, not causal explanations or calibrated confidence.

The design adapts established multi-branch/gating principles to this task; it is not a claim to invent gating. AI assisted with brainstorming, implementation, and analysis. The user selected the temporal-feature direction and requested a larger dataset. The student should review the results and make their own interpretation before submission.

## Data and relationship to HW2

- [Official UrbanSound8K documentation](https://urbansounddataset.weebly.com/urbansound8k.html)
- [Original archive, DOI 10.5281/zenodo.1203745](https://zenodo.org/records/1203745)
- J. Salamon, C. Jacoby and J. P. Bello, *A Dataset and Taxonomy for Urban Sound Research*, ACM Multimedia, 2014.
- 8,732 original clips (up to four seconds), compared with 240 in HW2: approximately 36.4 times as many. Ten classes instead of six. These are related environmental-sound tasks, not identical benchmarks. Do not compare HW3 percentages directly with HW2's 85.4%.
- The original archive is approximately 6 GB, and extracted data requires additional disk space. Archive MD5 is checked against `9aa69802bbf37fb986f71ec1483a196e`. Per-clip SHA-256, labels, source IDs and folds are recorded in `results/provenance.json`.
- Data and feature caches live in Git-ignored `data/raw/urbansound8k/` and `data/cache/`. Original license and Freesound attribution remain with the downloaded archive; no audio is republished by this HW3 experiment.

The baseline transfers the HW2 design, **168 → 64 ReLU → Dropout(0.5) → 10**, and retrains it on the new data. The ten-class head replaces the six-class head; pretrained HW2 weights are not reused. Like HW2, preprocessing is mono, 22,050 Hz, five seconds with zero padding, FFT 1024/hop 256, 64 log-Mel bands and 20 MFCCs, followed by temporal means and standard deviations. All official short clips are retained, including those below the web uploader's 0.25-second minimum. Feature standardization is fitted only on each training partition.

## Controlled designs

| ID | Design | Question |
|---|---|---|
| `baseline` | Global 168 → 64 → 10 | How does the HW2 design transfer? |
| `concat` | Concatenate global 168 + four ordered segments 672 → 64 → 10 | Are extra features sufficient? |
| `fixed` | Global and local branches each produce 64 units; average them → 10 | Does two-branch processing help without a learned gate? |
| `gated` | Same branches; scalar `a = sigmoid(W[g;l]+b)`; fuse `(1-a)g + a*l` → 10 | Does input-dependent routing help beyond fixed fusion? |

Each segment contains means/stds from the same full-clip spectrogram, in chronological order. Four equal groups of spectrogram frames yield 672 local features. The gate starts at 0.5. Fixed and gated variants use exactly identical initial global/local/output weights, batch order, and dropout random stream. Their difference is the 129 trainable gate parameters. The baseline has fewer parameters; compare `gated` against `fixed` to assess the gate without the large capacity change.

## Protocol

`config.json` was committed before training or inspecting test results. No test-driven tuning or discarded runs.

- Ten outer evaluations, holding out each official fold once. No random clip reshuffling.
- Test fold `k`, validation fold `k % 10 + 1`, train on the other eight folds. Source IDs are asserted disjoint across official folds.
- This uses all ten official test folds, but holds one additional fold out for validation rather than training on all nine non-test folds. It is a controlled homework experiment, not a directly comparable reproduction of published nine-fold training scores.
- Each of four variants trains for 80 epochs: 40 runs total, with identical examples, optimizer, epochs and batch size. The minimum-validation-cross-entropy checkpoint is restored before evaluation. The complete training budget is used even if validation loss rises.
- SGD, LR 0.03, momentum 0.9, weight decay 0.001, batch size 128, hidden width 64, dropout 0.5, CPU deterministic operations, two PyTorch threads.
- Seed `7400 + test_fold`, paired across variants. One seed per fold: fold variability is not a measure of initialization variability.
- Report accuracy, macro F1, loss, train/validation curves, fold SD, class recall, parameter counts, runtime and gates. Fold SD is descriptive, not a confidence interval; folds share training data.
- An additional diagnostic reverses the order of local segments at test time, retaining original positional scaling. This is a distribution shift, not definitive proof that chronological order is used; padding/position scaling can contribute. It does not affect checkpoint selection.

## Run

From the repository root, using the existing Python 3.11 environment:

```bash
python -m pip install -r requirements.txt
python -m training.hw3.prepare
python -m training.hw3.train
python -m training.hw3.analyze
python -m pytest tests/test_hw3.py -q
```

Download and extraction run once. Preparation verifies the archive and caches all features. Training saves each finished run independently and resumes completed runs only when their config hash matches. To rerun from scratch, move the prior `training/hw3/results/*-fold*.json` files and checkpoints into a local backup first. Do not mix new settings with old results. The executed notebook in this folder displays the model, training implementation and analysis.

`results/` contains all epoch histories, out-of-fold predictions, confusion matrices, and plots. `checkpoints/` contains selected state dictionaries and the corresponding training-only feature scaling for all runs. The separate report distinguishes the original hypothesis from observed results.
