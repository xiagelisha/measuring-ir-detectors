from pathlib import Path
import sys, json, csv, os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tmp/rewrite_vendor'))
R=ROOT/'doc/20260628/maijorRevise20261014'
OUT=R/'revision_r2_20261014'; A=OUT/'analysis'; F=OUT/'Figures';F.mkdir(exist_ok=True)
source=(R/'rewrite_v3/reproducibility/make_figures.py').read_text(encoding='utf8')
source=source.replace("A=Path(__file__).resolve().parents[1]/'source_data'", "A=ROOT/'doc/20260628/maijorRevise20261014/revision_r2_20261014/analysis/submitted_source/source_data'")
source=source.replace("OUT=Path(__file__).resolve().parents[1]/'figures'", "OUT=ROOT/'doc/20260628/maijorRevise20261014/revision_r2_20261014/Figures'")
source=source.replace("for ext in ['png','pdf','svg','tiff']", "for ext in ['png','pdf','svg']")
source=source.replace('Ten paired downstream seeds','Ten downstream seeds per configuration')
source=source.replace('Paired comparisons across initialisation and module choices','Primary comparison and recorded extensions')
source=source.replace("labels=['No module','Mean Teacher\\n(C2)','Metadata FiLM\\n(C3)','Colour-map stem\\n(C4)'];xs=[.25,.46,.67,.88]", "labels=['No module','Mean Teacher (C2)'];xs=[.35,.75]")
source=source.replace('The direction varies across seeds','Individual seed differences')
source=source.replace('Five-seed estimates span both signs','Overlapping subsets of the ten differences')
source=source.replace('Per-seed class trade-offs','Class changes co-vary (r = 0.712)')
source=source.replace('Run-level scores for all eight configurations','Historical records used as reporting examples')
source=source.replace('Paired effects within each initialisation','Historical differences within initialisation')
source=source.replace('Interaction with SSL initialisation','Historical interaction summaries')
source=source.replace("save(fig,'Fig5_interventions')", "save(fig,'FigS3_historical_reporting_examples')")
source=source.replace("ax.set_ylabel('Pairs for 80% power')", "ax.set_ylabel('Pairs for 80% power')\nax.secondary_yaxis('right',functions=(lambda x:2*x,lambda x:x/2)).set_ylabel('Total detector training runs')")
source=source.replace("fig.subplots_adjust(wspace=.42)", "fig.subplots_adjust(wspace=.72)")
source=source.replace("ax.text(12,.04,'10 seeds'", "ax.text(12,.04,'10 pairs = 20 runs'")
source=source.replace('Opposing means partly cancel','Class and aggregate estimates')
source=source.replace("ax.set_xlabel('Number of paired seeds')", "ax.set_xlabel('Number of seed pairs (two runs per pair)')")
sys.argv=[__file__,'--project-root',str(ROOT)]
exec(compile(source,'revised_base_figures','exec'))
# The base builder changes A and OUT to its own source and figure locations.
A=R/'revision_r2_20261014/analysis';OUT=F
U=json.loads((A/'uncertainty_summary.json').read_text())
rows=list(csv.DictReader((A/'per_run_test_uncertainty.csv').open()))
gr=list(csv.DictReader((A/'per_group_ap.csv').open()))
fig,axs=plt.subplots(2,2,figsize=(7.4,6.4));fig.subplots_adjust(hspace=.62,wspace=.55)
ax=axs[0,0]
for kind,off,col in [('Baseline',-.14,BLUE),('SSL',.14,ORANGE)]:
 rr=[r for r in rows if r['configuration']==kind];yy=np.array([float(r['mAP50']) for r in rr]);lo=np.array([float(r['mAP50_image_low']) for r in rr]);hi=np.array([float(r['mAP50_image_high']) for r in rr])
 ax.errorbar(np.arange(42,52)+off,yy,yerr=[yy-lo,hi-yy],fmt='o',ms=3,capsize=2,color=col,label=kind)
ax.set_xticks(range(42,52),range(42,52),rotation=45);ax.set_xlabel('Downstream seed');ax.set_ylabel('mAP50 and image bootstrap interval');ax.legend(frameon=False,fontsize=8);panel(ax,'a','Test-image uncertainty for each run')
ax=axs[0,1];keys=['image_only','seed_and_image_independent','seed_and_image_matched','cluster_only','seed_and_cluster_independent','seed_and_cluster_matched']
labs=['Images only','Seeds +\nimages','Matched seeds\n+ images','Groups only','Seeds +\ngroups','Matched seeds\n+ groups']
for i,k in enumerate(keys):
 lo,hi=U[k][0];v=U['point_difference'][0];col=BLUE if i<3 else ORANGE
 ax.plot([lo,hi],[i,i],color=col);ax.scatter([v],[i],color=col,s=25)
zero(ax,True);ax.set_yticks(range(6),labs,fontsize=8);ax.invert_yaxis();ax.set_xlabel('SSL − baseline mAP50 (95% interval)');panel(ax,'b','Separate and combined resampling')
for c,ax,letter in [('Water',axs[1,0],'c'),('Oil',axs[1,1],'d')]:
 groups=sorted(set(r['group'] for r in gr if r['class']==c))
 for k,g in enumerate(groups):
  for kind,off,col in [('Baseline',-.15,BLUE),('SSL',.15,ORANGE)]:
   vals=np.array([float(r['AP50']) for r in gr if r['group']==g and r['configuration']==kind]);xx=k+off
   ax.scatter(xx+np.linspace(-.04,.04,10),vals,color=col,s=12,alpha=.55);ax.scatter(xx,vals.mean(),marker='D',s=32,color=col)
 ax.set_xticks(range(2),[g.replace('_','\n',1)+f'\n(n = {next(r["images"] for r in gr if r["group"]==g)})' for g in groups],fontsize=8);ax.set_ylabel(f'{c} AP50');ax.set_ylim(0,1);panel(ax,letter,f'{c} results in the two test groups')
save(fig,'Fig5_test_uncertainty')
print('REVISED FIGURES COMPLETE')
