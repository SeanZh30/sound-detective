"""Build a transparent, executable experiment notebook without changing saved experiments."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
H=ROOT/'training/hw3'
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
cells=[md('''# HW3: Global statistics versus temporal fusion

**Question:** Does an input-dependent gate improve the HW2 MLP when ordered segment statistics are available?

UrbanSound8K has 8,732 recordings and 10 categories. We retrain the HW2 design on this dataset. Four variants and ten official test folds are fixed in `config.json` before the results are inspected. Eight nominal folds train; the next cyclic fold validates; the held-out fold tests. Five original source IDs cross official folds, so related train/validation clips are purged to isolate sources while preserving all test clips. This differs from training on nine folds without a separate validation set.

Run using the project Python environment. Existing run files with matching config hashes are reused; move them to a backup before an intentional complete retraining. The next cells expose the actual model and full optimization routine. AI assistance is documented in the README.'''),code('''from pathlib import Path
import os, sys, json
ROOT=Path.cwd().resolve()
ROOT=next(p for p in [ROOT,*ROOT.parents] if (p/'training/hw3/config.json').exists())
os.chdir(ROOT)
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
H=ROOT/'training/hw3'
cfg=json.loads((H/'config.json').read_text())
cfg'''),md('## 1. Features and source-separated data\nUse the same 168 global statistics as HW2, and four ordered segment vectors for the modified models. Short official clips are retained and padded to five seconds.'),code('''from training.hw3.prepare import main as prepare_data
prepare_data()
import numpy as np
cache=np.load(ROOT/'data/cache/hw3-features.npz')
print('Features:',cache['X'].shape)
print('Class counts:',np.bincount(cache['y']))
print('Official fold sizes:',np.bincount(cache['fold'])[1:])'''),md('## 2. Model definitions\nBaseline: global 168 → 64 → 10. Concat: 840 → 64 → 10. Fixed and gated: two 64-unit branches, fused before the output layer.'),code((H/'model.py').read_text()),md('## 3. Actual training implementation\nCross-entropy, backward propagation, SGD, paired batch order, validation checkpoint selection, and final test evaluation. Test results do not affect selection.'),code((H/'train.py').read_text().replace("HERE = Path(__file__).resolve().parent","HERE = ROOT / 'training/hw3'").split("if __name__ == '__main__':")[0]),code('main()'),md('## 4. Aggregate every fold and inspect diagnostics\nFold standard deviation describes variation across partitions, not an independent-trial confidence interval. The reversal diagnostic can reflect positional scaling and padding as well as temporal order.'),code('''from training.hw3.analyze import main as analyze
analyze()
summary=json.loads((H/'results/summary.json').read_text())
for name,m in summary['models'].items():
    print(f"{name:10s} accuracy={m['accuracy']['mean']:.3%} ± {m['accuracy']['std_across_folds']:.3%}, macro F1={m['macro_f1']['mean']:.4f}, parameters={m['parameters']}")
print('Paired differences:',summary['paired_accuracy_differences'])
print('Gate diagnostics:',summary['models']['gated']['gate'])'''),code('''from IPython.display import display,Image
for name in ['architecture','comparison','learning-curves','diagnostics']:
    display(Image(filename=str(H/'results'/f'{name}.png')))'''),md('## 5. Interpretation\nSee the experiment report and README for observations, alternative explanations and next steps. Distinguish the pre-experiment hypothesis above from the measurements below. No claim of improvement should be made solely from one fold or one favorable class.')]
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (Sound Detective)','language':'python','name':'python3'}})
nbf.write(nb,H/'03_temporal_gating.ipynb')
