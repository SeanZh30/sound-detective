"""Build teaching notebooks with real training cells, not opaque script runners."""
from pathlib import Path
import re
import textwrap
import nbformat as nbf
ROOT = Path(__file__).resolve().parents[1]
setup = '''from pathlib import Path
import os, sys
ROOT = Path.cwd().resolve()
ROOT = next((p for p in [ROOT, *ROOT.parents]
             if (p / "training/model.py").is_file() and (p / "frontend/package.json").is_file()), ROOT)
assert (ROOT / "training").exists(), "Open this notebook from the sound-detective repository."
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("Repository:", ROOT.name)
'''
metadata = {'kernelspec': {'display_name': 'Python 3 (Sound Detective)', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s.strip())
def save(name, cells):
    nb = nbf.v4.new_notebook(cells=cells, metadata=metadata)
    nbf.validate(nb)
    nbf.write(nb, ROOT / 'training/notebooks' / name)

data_cell = '''# Download the 240 selected recordings only if they are not already present.
from training.prepare import main as prepare_data
if not (ROOT / 'data/raw/manifest.json').exists():
    prepare_data()
'''
source = (ROOT / 'training/train.py').read_text()
header, body = source.split('def main():\n', 1)
body = textwrap.dedent(body.split("\nif __name__ == '__main__':")[0])
header = header.replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path.cwd()')
# The notebook visibly defines the MLP used by its training loop.
header = header.replace('from training.model import SoundMLP', (ROOT / 'training/model.py').read_text())
markers = ['    for hidden in [32, 64, 128]:', '    baseline_trials, baseline = [], None', '    model = SoundMLP(selected[\'hidden\'], selected[\'dropout\'])', "    torch.save({'state_dict': selected['state']"]
sections=[]
for marker in markers:
    before, body = body.split(textwrap.dedent(marker),1)
    sections.append(before)
    body=textwrap.dedent(marker)+body
sections.append(body)
cells=[md('''# HW2 — 01: Train a small MLP from scratch

逐步实现并训练六分类声音模型。这里的代码单元直接包含训练循环、反向传播和参数更新。

**Run all cells in order.** Use the project `.venv` kernel with `requirements-notebook.txt` installed. This notebook regenerates the MLP checkpoint and evaluation artifacts. It does not call `training.train.main()`.

Data: 240 ESC-50 clips; folds 1–3 train, 4 validation, 5 test. Our MLP learns from scratch. Notebook 02 fine-tunes a pretrained audio CNN on exactly the same split.'''),code(setup),code(header),md('## 1. Prepare data\nDownload only the selected classes. 同源录音不能跨训练、验证、测试集。'),code(data_cell),md('## 2. Features and split\n64 log-mel + 20 MFCC means/stds = 168 features. Scaling is fitted on training data only. The imported extractor is the same one used by the web API.'),code(sections[0]),md('## 3. Train the MLP\nOne hidden layer; cross-entropy loss; SGD with momentum. `backward()` computes gradients, then `step()` updates parameters. The declared 12-run search and early stopping use validation loss, never test performance.'),code(sections[1]),md('## 4. Shallow baseline\nLogistic regression receives the identical standardized features and split. Choose regularization using validation accuracy.'),code(sections[2]),md('## 5. Final evaluation\nRestore the validation-selected checkpoint and evaluate the set-aside test fold. This test set has been reported previously; it is a fixed homework comparison, not a new blind benchmark. Do not iterate based on its scores.'),code(sections[3]),md('## 6. Save weights and curves\nThe frontend and backend load these same saved artifacts. Restart the backend after retraining.'),code(sections[4]),code("from IPython.display import display, Image\ndisplay(Image(filename=str(OUT / 'learning-curves.png')))\nprint('Selected configuration:', report['config'])")]
save('01_mlp_from_scratch.ipynb', cells)
source=(ROOT/'training/finetune.py').read_text()
header,body=source.split('def main():\n',1)
body=textwrap.dedent(body.split("\nif __name__ == '__main__':")[0])
header=header.replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path.cwd()')
# Actual pretrained model definitions are included for review and execution.
header=header.replace('from training.pretrained import load_audioset, waveform32k, PRETRAIN_VERSION, PRETRAIN_MD5, PRETRAIN_URL', (ROOT/'training/pretrained.py').read_text().replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path.cwd()'))
parts=re.split(r'# SECTION: (\w+)\n',body)
notes={
'setup':'## 2. Load AudioSet weights and freeze layers\nThe original 527-way output is replaced with a randomly initialized six-way head. Conv blocks 1–3 and all BN running statistics stay frozen. Block 4 and the pretrained fc1 are genuinely updated, in addition to the new head. This is **partial fine-tuning**, not only a frozen embedding classifier.',
'cache':'## 3. Cache the frozen prefix\nCaching the first three frozen blocks avoids repeating unchanged computation. No dataset statistics are fitted here. The same waveform preprocessing is used at inference: 32 kHz, 5 seconds, 64 log-mel bands.',
'train':'## 4. Fine-tune pretrained layers\nFixed before viewing this model’s test result: AdamW, backbone LR 1e-4, head LR 1e-3, batch 8, max 20 epochs, patience 5. The explicit loop below computes logits, cross-entropy, gradients and parameter updates. Checkpoint selection uses validation loss only.',
'evaluate':'## 5. Restore and evaluate\nRestore the best validation checkpoint. Assertions check that the pretrained convolution and fc1 weights changed. Now evaluate the same held-out 48 clips used for the MLP.',
'save':'## 6. Save the fine-tuned checkpoint\nSave all weights, metrics, per-clip predictions, weight-change evidence and learning curves. The saved model can run offline without downloading the original AudioSet checkpoint.'}
cells=[md('''# HW2 — 02: Fine-tune a pretrained audio model

使用 **PANNs Cnn6**：它已在 AudioSet 上学习声音表示，我们再用六类 ESC-50 录音微调后面的层。

**Run notebook 01 first, then Run All here.** This notebook actually trains; all outputs below come from executed cells. CUDA/MPS is used when available, otherwise CPU. GPU runs can vary slightly across platforms.

We use the original AudioSet checkpoint, not a checkpoint already fine-tuned on ESC-50. This controlled homework comparison uses identical train/validation/test clips, but the pretrained model has additional external training data and a different architecture; it does not isolate the causal effect of pretraining. We have not independently audited all upstream pretraining data for overlap.

Sources: [PANNs official implementation](https://github.com/qiuqiangkong/audioset_tagging_cnn), [paper](https://arxiv.org/abs/1912.10211), [official weights](https://zenodo.org/records/3987831). MIT code license: `third_party/PANNs-LICENSE`.'''),code(setup),md('## 1. Architecture and verified pretrained weights\nAdapted Cnn6 implementation is shown below. The original checkpoint is verified against its published MD5 checksum before loading.'),code(header),code(data_cell)]
for i in range(1,len(parts),2):
    cells.extend([md(notes[parts[i]]),code(parts[i+1])])
cells.extend([code("from IPython.display import display, Image\ndisplay(Image(filename=str(OUT / 'pretrained-learning-curves.png')))"),md('## 7. Compare models\nCompare identical test clips. Do not assume a pretrained model must win. Interpret any difference alongside external pretraining, capacity, feature choice, and the small 48-clip test set.'),code('''simple = json.loads((OUT / 'metrics.json').read_text())
comparison = {
    'Logistic regression': simple['baseline']['test'],
    'MLP from scratch': simple['mlp']['test'],
    'Fine-tuned PANNs Cnn6': report['metrics']['test'],
}
for name, metrics in comparison.items():
    print(f"{name:26s} accuracy={metrics['accuracy']:.2%}, macro F1={metrics['macro_f1']:.4f}")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(list(comparison), [m['accuracy'] for m in comparison.values()], color=['gray', '#e57200', '#287ca0'])
ax.set(ylim=(0, 1), ylabel='Test accuracy', title='Same 48 held-out ESC-50 clips')
fig.tight_layout()
fig.savefig(OUT / 'model-comparison.png', dpi=160)
plt.show()
'''),md('## Explain it yourself\n预训练和微调有什么区别？哪些层更新了？冻结的 BN 为什么不能继续更新均值和方差？为什么两个模型不能仅凭准确率差就断言预训练一定更好？\n\nThe app can switch between both saved models. Probabilities from either model are not calibrated certainty.')])
save('02_pretrained_audio_finetuning.ipynb',cells)
print('Built two notebooks with explicit training cells.')
