"""Directly prune the joint four-condition circuit after factor composition.

Refinement of the factor-specific experiment. Fit/dev select every mask; the
same previously analyzed test/external populations evaluate the final graph.
"""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import concept_mapping.prune_neuron198_paths as cp

OUT=cp.OUT/'joint_refinement'


def main():
    OUT.mkdir(parents=True,exist_ok=True);began=time.monotonic();seed=50198
    with np.load(cp.SOURCE/'frozen_critic_weights.npz') as d:W=[d[f'W{j}'] for j in range(4)];b=[d[f'b{j}'] for j in range(4)]
    cp.EVAL_WEIGHTS=W
    with np.load(cp.BANK/'base_states.npz') as d:x=cp.branches(d['observations'])
    rows=cp.read(cp.OUT/'pruning_episode_split.csv')
    fit=np.array([i for i,r in enumerate(rows) if r['pruning_split']=='fit']);dev=np.array([i for i,r in enumerate(rows) if r['pruning_split']=='dev'])
    test=np.array([i for i,r in enumerate(rows) if r['pruning_split']=='heldout'])
    full=cp.forward(x,W,b)
    with np.load(cp.SOURCE/'fresh_base_states.npz') as d:ex=cp.branches(d['observations'])
    er=cp.read(cp.SOURCE/'fresh_state_selection.csv');ef=cp.forward(ex,W,b)
    protocol=dict(hypothesis='Jointly selected paths can restore and remove the four-condition target response from the neither-state baseline.',
        motivation='The separately selected diamond and pickaxe paths recovered 86.5% when combined, below the 90% target. This is a refinement, not an untouched replication.',
        background='Actual same-state neither branch for restoration; actual factual branch for necessity. No discovery means.',
        criteria=json.loads((cp.OUT/'protocol.json').read_text())['criteria'],
        selection='Exactly the same fit/dev episodes as factor-specific pruning. No held-out/external effect chooses masks.',
        full_context_scope='Unselected messages retain original matched state activity. Only the routed input-to-198 paths are explained.',
        no_training_or_rollouts=True,seed=seed)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');history=[]
    fm=cp.PathModel(full,W,b,fit,'joint');dm=cp.PathModel(full,W,b,dev,'joint')
    mask,node_info=cp.prune(fm,dm,seed,'joint_L2_to_198',history);parents=np.flatnonzero(mask)
    print('Joint L2 parents:',parents.tolist(),flush=True)
    fm=cp.PathModel(full,W,b,fit,'joint',parents);dm=cp.PathModel(full,W,b,dev,'joint',parents)
    assert cp.passes(dm.metric())
    mask,edge_info=cp.prune(fm,dm,seed+1,'joint_L1_to_L2',history)
    live=mask.any(1);parents=parents[live];mask=mask[live];l1=np.flatnonzero(mask.any(0))
    source=np.flatnonzero(np.any(x[:,np.r_[fit,dev]]!=x[2,np.r_[fit,dev]][None],axis=(0,1)))
    circuit=dict(L2_parents=parents.tolist(),L1_parents=l1.tolist(),input_coordinates=source.tolist(),
        input_edges=len(source)*len(l1),L1_to_L2_edges=int(mask.sum()),L2_to_198_edges=len(parents),output_edges=1,
        node_pruning=node_info,edge_pruning=edge_info)
    (OUT/'selection.json').write_text(json.dumps(circuit,indent=2)+'\n')
    np.savez_compressed(OUT/'path_mask.npz',L2_parents=parents,L1_to_L2_mask=mask,L1_parents=l1,input_coordinates=source)
    selected_edges=[]
    for src in source:
        for dest in l1:selected_edges.append(dict(layer=1,source=int(src),target=int(dest),weight=float(W[0][src,dest])))
    for j,src in np.argwhere(mask):selected_edges.append(dict(layer=2,source=int(src),target=int(parents[j]),weight=float(W[1][src,parents[j]])))
    for src in parents:selected_edges.append(dict(layer=3,source=int(src),target=198,weight=float(W[2][src,198])))
    selected_edges.append(dict(layer=4,source=198,target=0,weight=float(W[3][198,0])))
    cp.write(OUT/'selected_edges.csv',selected_edges)
    cp.write(OUT/'pruning_history.csv',history)
    results=[];details={};strata=[]
    for name,ff,rr,idx in [('fit',full,rows,fit),('dev',full,rows,dev),('heldout',full,rows,test),('external',ef,er,np.arange(len(er)))]:
        model=cp.PathModel(ff,W,b,idx,'joint',parents,mask)
        summary,records=cp.evaluate(model,ff,idx,'joint',rr,10000,seed)
        summary['population']=name;results.append(summary);details[name]=records;cp.write(OUT/f'{name}_per_state.csv',records)
        print(name,summary,flush=True)
        if name=='heldout':
            for group in sorted({rr[i]['stratum'] for i in idx}):
                ii=np.array([i for i in idx if rr[i]['stratum']==group]);m=cp.PathModel(ff,W,b,ii,'joint',parents,mask).metric()
                strata.append(dict(stratum=group,n_states=len(ii),**m))
    cp.write(OUT/'summary.csv',results);cp.write(OUT/'heldout_strata.csv',strata)
    rng=np.random.default_rng(seed+99);random=[]
    for draw in range(32):
        rp=rng.choice(512,len(parents),replace=False);rm=np.zeros_like(mask)
        for j in range(len(rp)):rm[j,rng.choice(512,int(mask[j].sum()),replace=False)]=True
        m=cp.PathModel(full,W,b,test,'joint',rp,rm).metric();random.append(dict(draw=draw,**m))
    cp.write(OUT/'random_controls.csv',random)
    # Reuse the five outcome-independent held-out examples from the factor test.
    example_rows=[r for r in cp.read(cp.OUT/'example_selection.csv') if r['factor']=='diamond']
    examples(OUT,x,details['heldout'],example_rows)
    plot(OUT,results,parents,W,full,test,details['heldout']);diagram(OUT,circuit,W)
    report=dict(protocol=protocol,circuit=circuit,results=results,heldout_strata=strata,
        random_controls=random,elapsed_seconds=time.monotonic()-began,
        limitations='One fixed critic; adaptive refinement on previously inspected evaluation populations. Conditional path circuit with untraced surrounding context; greedy masks are not globally minimal.')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');write_report(OUT,report)
    print(json.dumps(dict(circuit=circuit,results=results,elapsed=report['elapsed_seconds']),indent=2),flush=True)


def plot(out,results,parents,W,full,test,records):
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);rr=[r for r in results if r['population'] in ['heldout','external']]
    xx=np.arange(2)
    err=np.array([[r['recovery_pct']-r['recovery_ci_low'] for r in rr],[r['recovery_ci_high']-r['recovery_pct'] for r in rr]])
    axes[0].bar(xx-.16,[r['recovery_pct'] for r in rr],width=.3,yerr=err,capsize=3,label='Paths restored (95% CI)',color='#2563eb')
    axes[0].bar(xx+.16,[r['residual_pct'] for r in rr],width=.3,label='Paths removed',color='#dc2626')
    axes[0].axhline(100,color='black',lw=.8);axes[0].set_xticks(xx,[r['population'] for r in rr]);axes[0].set_ylabel('Original neuron-198 interaction (%)');axes[0].set_title('Joint necessity and recovery');axes[0].legend(frameon=False,prop={'size':8,'weight':'bold'})
    matrix=np.array([[np.mean([r[f'h198_{key}{suffix}'] for r in records]) for key in ['A','B','D','C']] for suffix in ['', '_restored','_removed']])
    pic=axes[1].imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto');axes[1].set_xticks(range(4),['Both','Diamond','Pickaxe','Neither'],rotation=20,ha='right');axes[1].set_yticks(range(3),['Original','Restored','Removed']);axes[1].set_title('Held-out target response')
    for i in range(3):
        for j in range(4):axes[1].text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontweight='bold',color='white' if abs(matrix[i,j])>.6 else 'black')
    fig.colorbar(pic,ax=axes[1],shrink=.8,label='Mean activation');cp.save(fig,out,'joint_circuit_results')


def diagram(out,c,W):
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    with np.load(out/'path_mask.npz') as d:mask=d['L1_to_L2_mask'];parents=d['L2_parents']
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True);ax.set(xlim=(0,14.8),ylim=(-.5,7.7));ax.axis('off')
    def box(x,y,w,h,text,color='#eff6ff',size=9):
        ax.add_patch(FancyBboxPatch((x,y-h/2),w,h,boxstyle='round,pad=.06',facecolor=color,edgecolor='#475569'))
        ax.text(x+w/2,y,text,ha='center',va='center',fontsize=size,fontweight='bold')
    def arrow(a,b,color='#64748b',lw=1.3):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=color,lw=lw))
    box(.1,5.3,2,1.15,'Diamond / stone\n124 input slots')
    box(.1,1.9,2,1.15,'Iron pickaxe\nInput slot 1331',color='#f3e8ff')
    box(2.9,3.6,2.4,1.6,f'Hidden layer 1\n{len(c["L1_parents"])} selected units\nState-dependent context')
    arrow((2.2,5.3),(2.8,4.0),'#2563eb');arrow((2.2,1.9),(2.8,3.2),'#9333ea')
    ys=dict(zip([159,334,461,195,236,256],[6.65,5.5,4.35,3.2,2.05,.9]))
    for parent in [159,334,461,195,236,256]:
        y=ys[parent];box(6.3,y,1.2,.65,str(parent),size=10)
        arrow((5.4,3.6),(6.2,y))
        count=int(mask[list(parents).index(parent)].sum())
        ax.text(5.3,.55*y+1.67,f'{count} edges',fontsize=7,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',pad=.4))
        w=float(W[2][parent,198]);arrow((7.6,y),(9.4,3.6),'#dc2626' if w<0 else '#2563eb',1+abs(w))
        ax.text(8.0,.6*y+1.47,f'{w:+.3f}',fontsize=8,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',pad=.4))
    box(9.5,3.6,2.1,1.05,'Neuron 198\ntanh',color='#fef3c7',size=11)
    box(12.5,3.6,2,1.0,'Scalar value',color='#fef3c7',size=10)
    arrow((11.7,3.6),(12.4,3.6));ax.text(11.65,4.3,f'{float(W[3][198,0]):+.4f}',fontsize=9,fontweight='bold')
    ax.set_title('Joint conditional circuit for diamond–pickaxe valuation',fontweight='bold')
    ax.text(.1,-.14,f'{c["L1_to_L2_edges"]} L1→L2 connections; {c["input_edges"]:,} input messages. Original weights retained.\n'
        'Surrounding matched context remains active; input messages were not individually minimized.',fontsize=9,fontweight='bold')
    cp.save(fig,out,'joint_circuit_diagram')


def examples(out,x,records,selection):
    from PIL import Image,ImageDraw,ImageFont,JpegImagePlugin,PdfImagePlugin
    from craftax.craftax_classic.constants import load_all_textures
    from concept_mapping.visualize_symbolic_trajectory import render_local_view
    dest=out/'examples';dest.mkdir(exist_ok=True);lookup={r['state_index']:r for r in records}
    textures=load_all_textures(7);pt=load_all_textures(28);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',17);sheets=[]
    for number,s in enumerate(selection,1):
        i=int(s['state_index']);r=lookup[i];panels=[]
        for c,key in zip([0,1,3,2],['A','B','D','C']):
            view=render_local_view(x[c,i],block_size=7,include_inventory=True,display_scale=4,textures=textures,panel_textures=pt)
            panel=Image.new('RGB',(view.width,view.height+114),'white');panel.paste(view,(0,0));draw=ImageDraw.Draw(panel)
            labels=[f'value: {r[f"V_{key}"]:.4f} → {r[f"V_{key}_removed"]:.4f}',f'h198 original: {r[f"h198_{key}"]:+.3f}',
                f'h198 removed: {r[f"h198_{key}_removed"]:+.3f}',f'h198 restored: {r[f"h198_{key}_restored"]:+.3f}']
            for line,text in enumerate(labels):draw.text((3,view.height+5+25*line),text,font=font,fill='black' if line==0 else '#1d4ed8')
            panels.append(panel)
        w,h=panels[0].size;sheet=Image.new('RGB',(2*w+18,2*h+18),'white')
        for j,panel in enumerate(panels):sheet.paste(panel,((j%2)*(w+18),(j//2)*(h+18)))
        sheet.save(dest/f'example_{number:02d}.png');sheets.append(sheet)
    sheets[0].save(dest/'all_examples.pdf',save_all=True,append_images=sheets[1:]);cp.write(dest/'selection.csv',selection)


def write_report(out,r):
    c=r['circuit'];lines=['# Joint four-condition refinement of neuron-198 path pruning','',r['protocol']['motivation'],'',
        'Masks are selected on the same fit/dev discovery episodes. The held-out and external evaluations have been inspected in earlier iterations; this is not a fresh replication. '
        'All context outside the selected paths remains supplied by matched original-model states. Weights are frozen.','',
        f'Selected L2 parents: **{c["L2_parents"]}**. '
        f'{len(c["L1_parents"])} selected L1 neurons, {c["L1_to_L2_edges"]} L1→L2 messages, {c["L2_to_198_edges"]} messages into 198. '
        f'{c["input_edges"]} original input→L1 messages carry the 125 changed input coordinates; these input edges were not individually minimized.','',
        '| Population | States | Recovery | Residual after removal | Recovery error | Activation MAE | All fixed criteria pass |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for x in r['results']:lines.append(f'| {x["population"]} | {x["n_states"]} | {x["recovery_pct"]:.2f}% | {x["residual_pct"]:.2f}% | {x["recovery_error"]:.3f} | {x["activation_mae"]:.3f} | {x["passes"]} |')
    lines.extend(['','Percentages concern the original activation interaction of neuron 198. They do not indicate the fraction of all critic computation explained. '
        'Confidence intervals use 10,000 episode-bootstrap draws. All four counterfactuals and all states from an episode remain together.','',
        '## Interpretation','',
        'The six selected L2 inputs are 159/195/236/256/334/461. The joint search retains 317 L1→L2 connections through 163 L1 neurons. '
        'It restores 92.17% of held-out target interaction (95% episode-bootstrap CI 91.51–92.84%), while blocking it leaves -0.084% '
        'of the mean target interaction. Removal changes the entire critic’s interaction by 24.07%, consistent with intervening on only one final-hidden unit.','',
        'This is a causally important partial circuit with supporting original-model context. It narrowly misses the complete prespecified '
        'held-out criteria: normalized recovery error is 0.1048 versus a 0.10 limit, and activation MAE is 0.0531 versus 0.05. '
        'The external mean recovery is 101.23%, but normalized per-state error is 25.11%; a matching mean does not establish individual-state fidelity. '
        'Therefore the study does not establish a complete, universally sufficient AND gate.','',
        '- [Joint circuit diagram](joint_circuit_diagram.pdf)', '- [Joint results](joint_circuit_results.pdf)', '- [Five examples](examples/all_examples.pdf)',
        '- [Selected mask](selection.json), [pruning trajectory](pruning_history.csv), [held-out distance groups](heldout_strata.csv)',
        '- [Full results and intervals](summary.json), [random controls](random_controls.csv)','',r['limitations'],'',
        'Reproduce in jax-v100-cuda12:','', '```bash',
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/mpl-causal-joint python concept_mapping/prune_neuron198_joint.py',
        '```',''])
    (out/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
