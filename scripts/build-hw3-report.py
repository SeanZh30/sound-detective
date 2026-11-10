"""Render the two-page HW3 summary from the measured experiment artifacts.
Run with Python + reportlab; training dependencies remain separate.
"""
import json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
ROOT=Path(__file__).resolve().parents[1]
H=ROOT/'training/hw3'; OUT=H/'results'
s=json.loads((OUT/'summary.json').read_text());models=s['models'];paired=s['paired_accuracy_differences']
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCompact',fontName='Helvetica',fontSize=9.3,leading=12,spaceAfter=7))
styles.add(ParagraphStyle(name='SectionCompact',fontName='Helvetica-Bold',fontSize=12,leading=15,spaceBefore=9,spaceAfter=5,textColor=colors.HexColor('#493971')))
styles.add(ParagraphStyle(name='SmallNote',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=6,textColor=colors.HexColor('#555566')))
story=[]
def p(text,style='BodyCompact'): story.append(Paragraph(text,styles[style]))
def section(text):p(text,'SectionCompact')
p('Does temporal information need a learned gate?','Title')
p('DS7400 HW3 | Global and segment features for sound classification','SmallNote')
section('1. What I wanted to change')
p('My HW2 MLP used whole-clip means and standard deviations. These features describe the overall sound, but do not distinguish an event at the beginning from the same event at the end. I tested whether four ordered segments and a learned mixing weight would help. The hypothesis was recorded before training: the gate could favor local features when they were useful and reduce their contribution otherwise. I expected possible problems from overfitting, gate saturation, fixed segment boundaries and zero padding. The design uses established gating ideas; the contribution here is the feature combination and its evaluation on this task.')
section('2. Data, model and comparisons')
p('UrbanSound8K contains 8,732 clips in 10 classes (36.4 times the HW2 sample count). HW2 used 240 clips from a six-class ESC-50 subset. Changing the dataset and labels means this is a transfer of the HW2 design, not a comparison with its original accuracy. The baseline retrains the HW2 architecture with a 10-class head: 168 global features, 64 ReLU units, dropout 0.5, and cross-entropy loss. Audio is mono, 22,050 Hz, padded to 5 seconds; 64 log-Mel and 20 MFCC temporal means/stds form the 168 features. All official short clips are retained.')
p('For each of ten official test folds, the next cyclic fold validates and the other eight train. An audit found five source IDs across folds (152 clips). Test sources are purged from validation; test and retained validation sources are purged from training. All test clips remain. Scaling uses training data only. All four models train for 80 epochs with SGD (LR 0.03, momentum 0.9, weight decay 0.001, batch 128); the lowest-validation-loss checkpoint is restored. Seeds are paired across models (7400 + fold). This is ten-fold outer evaluation with separate validation, not the published protocol that trains on all nine non-test folds. No test-driven tuning was performed.')
p('<b>Proposed mapping:</b> global 168 -&gt; 64 ReLU = g; local 672 -&gt; 64 ReLU = l; a = sigmoid(Linear([g;l])); h = (1-a)g + a·l; dropout(h) -&gt; 10 logits. The four local feature groups are computed from consecutive quarters of the padded spectrogram; each contributes 168 statistics. The gate produces one weight per recording, not one per segment.')
p('Controls: concatenation feeds all 840 features to one 64-unit layer; fixed fusion uses the same two branches with a = 0.5. Fixed and gated start from identical branch/head weights, batch order and dropout stream. Only 129 gate parameters differ. The baseline is much smaller, so gated-versus-fixed is the cleaner test of routing.','SmallNote')
section('3. Results')
rows=[['Model','Accuracy ± fold SD','Macro F1','Parameters','Seconds/run']]
names={'baseline':'HW2-style MLP','concat':'Concatenation','fixed':'Fixed fusion','gated':'Learned gate'}
for v,m in models.items():rows.append([names[v],f"{m['accuracy']['mean']*100:.2f}% ± {m['accuracy']['std_across_folds']*100:.2f}",f"{m['macro_f1']['mean']:.3f}",f"{m['parameters']:,}",f"{m['mean_training_seconds']:.1f}"])
t=Table(rows,colWidths=[113,144,73,98,112]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeafa')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#ddd8e8'))]));story.append(t)
p('Accuracy and F1 are unweighted means over folds. SD describes partition variability, not a confidence interval. Timing includes training/evaluation with four fold workers sharing the CPU; it is not an isolated speed benchmark.','SmallNote')
story.append(PageBreak())
p('Why the learned gate fell short','Title')
g=models['gated']['gate']
section('4. Did the gate help?')
for ref in ['baseline','fixed','concat']:
 d=paired['gated_minus_'+ref]
 p(f"The learned gate differs from <b>{names[ref]}</b> by <b>{d['mean']*100:+.2f} percentage points</b> on average, with higher accuracy in {d['positive_folds']} of 10 folds. The fold-paired difference SD is {d['std']*100:.2f} points.")
best=max(models,key=lambda v:models[v]['accuracy']['mean'])
if best!='gated':
 p(f"<b>The proposed gate is not the strongest design:</b> {names[best]} has the highest mean accuracy. The result does not support my expectation that a learned weight would beat equal weighting. The gain over the small baseline cannot be credited to the gate alone: local features, extra parameters and the branch structure all change as well.")
else:
 p('The proposed gate has the highest mean accuracy in this fixed experiment. This supports further investigation, but does not prove a robust improvement: training partitions overlap, only one seed is used per fold, and larger models have more capacity than the baseline.')
section('5. Possible explanations')
p(f"The local-branch gate has pooled mean {g['mean']:.3f}, SD {g['std']:.3f}, and range {g['min']:.3f}-{g['max']:.3f}. {100*g['saturation_fraction']:.1f}% of clips have gates below 0.05 or above 0.95. For many clips, the model therefore relies mostly on one branch. Small sigmoid derivatives near the endpoints may make the gate harder to adjust. This is a possible explanation, not a demonstrated cause; padding and background differences may also affect the weights.")
rev=models['gated']['reversed_accuracy_mean']
p(f"Reversing segment order at evaluation changes gated accuracy from {models['gated']['accuracy']['mean']*100:.2f}% to {rev*100:.2f}%. This was a diagnostic, not a model-selection step. Reversal also moves padded regions and applies the original position-specific scaling to shifted features. The drop therefore does not show that the model learned meaningful event order.")
# Describe convergence and generalization at the selected checkpoints using actual measurements.
rs=[json.loads((OUT/f'gated-fold{f}.json').read_text()) for f in range(1,11)]
train=sum(r['metrics']['train']['accuracy'] for r in rs)/10
val=sum(r['metrics']['validation']['accuracy'] for r in rs)/10
first=sum(r['history'][0]['train_loss'] for r in rs)/10
last=sum(r['history'][-1]['train_loss'] for r in rs)/10
p(f"Mean gated training cross-entropy falls from {first:.3f} after epoch 1 to {last:.3f} after epoch 80. Selected-checkpoint train/validation accuracy averages {100*train:.2f}%/{100*val:.2f}%; this gap suggests generalization limitations. No nonfinite-loss failure occurred. Selecting the lowest validation-loss checkpoint does not eliminate this generalization gap.")
story.append(Image(str(OUT/'comparison.png'),width=540,height=196))
section('6. What I would test next')
p('The fixed-fusion model was the best of the four designs. For this setup, adding a learned mixing weight was not worth its extra flexibility. I would next test a gentler sigmoid gate and divide clips by their actual duration rather than their padded length. A global-only model with a similar parameter count would help separate the effect of temporal features from capacity. Repeating several seeds would also matter: this experiment used only one seed per fold, and the folds share training examples. These are follow-up experiments, not results already obtained. Any new settings should be chosen using validation data.')
p('Reference: J. Salamon, C. Jacoby & J. P. Bello, A Dataset and Taxonomy for Urban Sound Research, ACM Multimedia (2014). Dataset: doi.org/10.5281/zenodo.1203745. AI assistance: Codex helped with brainstorming, code, debugging, analysis and report drafting.','SmallNote')
def footer(canvas,doc):
 canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#777788'));canvas.drawString(36,22,'Sound Detective | HW3 experiment summary');canvas.drawRightString(576,22,str(doc.page))
SimpleDocTemplate(str(H/'report.pdf'),pagesize=(612,792),rightMargin=36,leftMargin=36,topMargin=30,bottomMargin=32).build(story,onFirstPage=footer,onLaterPages=footer)
print(H/'report.pdf')
