"""Summarize every predeclared run; paired folds are descriptive, not independent trials."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
LABELS=['air conditioner','car horn','children playing','dog bark','drilling','engine idling','gun shot','jackhammer','siren','street music']
NAMES={'baseline':'HW2 MLP','concat':'Concatenation','fixed':'Fixed fusion','gated':'Learned gate'}
COLORS={'baseline':'#68758a','concat':'#188c87','fixed':'#b08032','gated':'#6652b8'}

def main():
    cfg=json.loads((HERE/'config.json').read_text())
    reports={v:[json.loads((OUT/f'{v}-fold{f}.json').read_text()) for f in cfg['test_folds']] for v in cfg['models']}
    summary={'protocol':cfg,'models':{},'paired_accuracy_differences':{}}
    for v,runs in reports.items():
        metrics={}
        for key in ['accuracy','macro_f1','loss']:
            values=np.array([r['metrics']['test'][key] for r in runs])
            metrics[key]={'mean':float(values.mean()),'std_across_folds':float(values.std(ddof=1)),'folds':values.tolist()}
        predictions=[p for r in runs for p in r['test_predictions']]
        assert len(predictions)==8732 and len({p['filename'] for p in predictions})==8732
        cm=np.sum([r['metrics']['test']['confusion_matrix'] for r in runs],axis=0)
        metrics.update({'parameters':runs[0]['parameters'],'mean_training_seconds':float(np.mean([r['seconds'] for r in runs])),
                       'best_epochs':[r['best_epoch'] for r in runs], 'confusion_matrix':cm.tolist(),
                       'class_recall':(cm.diagonal()/cm.sum(1)).tolist(),
                       'pooled_accuracy':float(sum(p['actual']==p['predicted'] for p in predictions)/8732)})
        if v!='baseline':
            metrics['reversed_accuracy_mean']=float(np.mean([r['reversed_segments_test']['accuracy'] for r in runs]))
        if v=='gated':
            gates=np.array([p['local_gate'] for p in predictions])
            metrics['gate']={'mean':float(gates.mean()),'std':float(gates.std()),'min':float(gates.min()),'max':float(gates.max()),
                'saturation_fraction':float(((gates<.05)|(gates>.95)).mean()),
                'class_means':[float(np.mean([p['local_gate'] for p in predictions if p['actual']==c])) for c in range(10)]}
        summary['models'][v]=metrics
    for reference in ['baseline','concat','fixed']:
        d=np.array(summary['models']['gated']['accuracy']['folds'])-np.array(summary['models'][reference]['accuracy']['folds'])
        summary['paired_accuracy_differences']['gated_minus_'+reference]={'mean':float(d.mean()),'std':float(d.std(ddof=1)),
             'positive_folds':int((d>0).sum()),'folds':d.tolist()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for v in cfg['models']:
        axes[0].plot(range(1,11),np.array(summary['models'][v]['accuracy']['folds'])*100,'o-',label=NAMES[v],color=COLORS[v])
    axes[0].set(xlabel='Official test fold',ylabel='Test accuracy (%)',xticks=range(1,11));axes[0].legend(fontsize=8)
    vals=[summary['models'][v]['accuracy']['mean']*100 for v in cfg['models']]
    std=[summary['models'][v]['accuracy']['std_across_folds']*100 for v in cfg['models']]
    axes[1].bar([NAMES[v] for v in cfg['models']],vals,yerr=std,capsize=4,color=[COLORS[v] for v in cfg['models']])
    axes[1].set(ylabel='Mean accuracy (%)',title='Error bars: SD across folds');axes[1].tick_params(axis='x',labelrotation=15)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for ax,(v,runs) in zip(axes.ravel(),reports.items()):
        for split,style in [('train','-'),('validation','--')]:
            values=np.array([[e[split+'_loss'] for e in r['history']] for r in runs])
            mean=values.mean(0);sd=values.std(0)
            ax.plot(range(1,81),mean,style,label=split,color=COLORS[v] if split=='train' else '#b14a52')
            ax.fill_between(range(1,81),mean-sd,mean+sd,alpha=.12)
        ax.set(title=NAMES[v],xlabel='Epoch',ylabel='Cross-entropy');ax.legend()
    fig.suptitle('All 10 folds: mean loss and fold SD (80 epochs per run)');fig.tight_layout();fig.savefig(OUT/'learning-curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    g=summary['models']['gated']['gate']
    pred=[p for r in reports['gated'] for p in r['test_predictions']]
    axes[0].hist([p['local_gate'] for p in pred],bins=25,color=COLORS['gated']);axes[0].set(xlabel='Local-branch gate (all out-of-fold clips)',ylabel='Clip count',xlim=(0,1))
    for v in ['baseline','gated']:
        axes[1].plot(np.array(summary['models'][v]['class_recall'])*100,range(10),'o-',label=NAMES[v],color=COLORS[v])
    axes[1].set(yticks=range(10),yticklabels=LABELS,xlabel='Pooled out-of-fold recall (%)');axes[1].legend()
    fig.tight_layout();fig.savefig(OUT/'diagnostics.png',dpi=160);plt.close(fig)
    # Architecture figure used by the notebook, report and demo.
    fig,ax=plt.subplots(figsize=(11,4));ax.axis('off')
    boxes=[(.03,.55,.20,.20,'Global statistics\n168 → 64 ReLU'),(.03,.12,.20,.20,'4 ordered segments\n672 → 64 ReLU'),(.37,.32,.23,.22,'a = sigmoid(Linear([g,l]))\n64 + 64 → 1'),(.70,.32,.27,.22,'(1-a)g + a·l\nDropout 0.5 → Linear 10')]
    from matplotlib.patches import FancyBboxPatch
    for x,y,w,h,t in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',fc='#eeeafa',ec='#6652b8'))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11)
    for a,b in [((.25,.65),(.35,.47)),((.25,.22),(.35,.38)),((.62,.43),(.68,.43))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.5})
    ax.text(.5,.91,'Proposed design: input-dependent global / temporal fusion',ha='center',fontsize=15,weight='bold')
    ax.text(.5,.02,'Baseline: global 168 → 64 ReLU → Dropout → 10     |     Fixed ablation: a = 0.5',ha='center',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'architecture.png',dpi=170);plt.close(fig)
    print(json.dumps({v:{'accuracy':m['accuracy']['mean'],'f1':m['macro_f1']['mean'],'parameters':m['parameters']} for v,m in summary['models'].items()},indent=2))

if __name__=='__main__':
    main()
