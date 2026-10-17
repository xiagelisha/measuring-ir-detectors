from pathlib import Path
import sys,json,csv,itertools,os
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'mpl_cache')
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'tmp/rewrite_vendor'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from statsmodels.stats.power import TTestPower
from PIL import Image
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[5])
args=parser.parse_args()
ROOT=args.project_root.resolve()
A=Path(__file__).resolve().parents[1]/'source_data'
OUT=Path(__file__).resolve().parents[1]/'figures';OUT.mkdir(exist_ok=True)
D=json.loads((A/'data.json').read_text());S=json.loads((A/'stats.json').read_text())
BLUE='#2878A5';ORANGE='#D9782D';GREEN='#26866D';RED='#B54A50';GRAY='#65717B';PURPLE='#7C64A4'
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,'axes.titleweight':'bold','axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'lines.linewidth':1.3,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def arr(k,m='mAP50'):return np.array([r[m] for r in D[k]])
def ci(x):
 x=np.asarray(x); m=x.mean(); h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return m,h
def panel(ax,letter,title):
 ax.set_title(title,loc='left',pad=13)
 ax.text(-.10,1.065,letter,transform=ax.transAxes,fontweight='bold',fontsize=12,va='bottom')
def save(fig,name):
 from matplotlib.lines import Line2D
 from matplotlib.collections import Collection
 from matplotlib.text import Text
 for artist in fig.findobj():
  if isinstance(artist,Line2D):
   artist.set_linewidth(max(1.4,artist.get_linewidth()))
   artist.set_markeredgewidth(max(1.4,artist.get_markeredgewidth()))
  elif isinstance(artist,Collection):
   widths=artist.get_linewidths()
   if len(widths):artist.set_linewidths([max(1.4,w) for w in widths])
  elif isinstance(artist,Text) and artist.get_text():artist.set_fontsize(max(8,artist.get_fontsize()))
 for ax in fig.axes:
  ax.tick_params(width=1.4)
  for spine in ax.spines.values():spine.set_linewidth(1.4)
 for ext in ['png','pdf','svg','tiff']:fig.savefig(OUT/f'{name}.{ext}',dpi=450,bbox_inches='tight',pad_inches=.12)
 plt.close(fig)
def zero(ax,vertical=False):
 (ax.axvline if vertical else ax.axhline)(0,color='#91979B',linestyle='--',lw=.8,zorder=0)

# Figure 1: actual images, split evidence and study design.
fig=plt.figure(figsize=(7.2,6.1));gs=fig.add_gridspec(3,2,height_ratios=[1.1,1,1.25],hspace=.65,wspace=.45)
meta=list(csv.DictReader((ROOT/'data/metadata.csv').open(encoding='utf-8-sig')));tests=set((ROOT/'data/splits/main_holdout/test.txt').read_text().splitlines())
examples=[]
for idx,(key,label,color) in enumerate([('has_water','Water',BLUE),('has_oil','Oil',ORANGE)]):
 candidates=[r for r in meta if r['image_name'] in tests and int(r[key]) and 1<=int(r['n_boxes'])<=5]
 r=candidates[len(candidates)//2];examples.append(r['image_name'])
 ax=fig.add_subplot(gs[0,idx]);im=Image.open(ROOT/r['image_path']);ax.imshow(im);w,h=im.size
 for l in (ROOT/r['label_path']).read_text().splitlines():
  c,x,y,bw,bh=map(float,l.split());ax.add_patch(Rectangle(((x-bw/2)*w,(y-bh/2)*h),bw*w,bh*h,fill=False,edgecolor=BLUE if c==0 else ORANGE,lw=1.1))
 ax.set_axis_off();panel(ax,'ab'[idx],f'{label} annotations in a test image')
ax=fig.add_subplot(gs[1,0]);ss=['train','val','test'];water=[S['splits'][s]['water_boxes'] for s in ss];oil=[S['splits'][s]['oil_boxes'] for s in ss]
ax.barh(range(3),water,color=BLUE,label='Water');ax.barh(range(3),oil,left=water,color=ORANGE,label='Oil')
ax.set_yticks(range(3),['Training','Validation','Test']);ax.invert_yaxis();ax.set_xlabel('Annotated objects');ax.set_xlim(0,20500)
for i,s in enumerate(ss):ax.text(water[i]+oil[i]+180,i,f"{len(S['splits'][s]['sessions'])} groups",va='center',fontsize=7)
ax.legend(frameon=False,ncol=2,fontsize=7,loc='lower right');panel(ax,'c','Object counts by split')
ax=fig.add_subplot(gs[1,1]);ax.axis('off');panel(ax,'d','Fixed evaluation design')
for i,(head,body) in enumerate([('2,593 images • 22 sequence groups','1,886 training / 348 validation / 359 test'),('SSL sees training and validation images','3,920 frame pairs • test groups excluded'),('Ten paired downstream seeds','42–51 • one fixed split and SSL checkpoint')]):
 y=.93-i*.34;ax.text(0,y,head,fontweight='bold',fontsize=8.5,va='top');ax.text(0,y-.13,body,fontsize=7.7,va='top',color=GRAY)
ax=fig.add_subplot(gs[2,:]);ax.axis('off');panel(ax,'e','Paired comparisons across initialisation and module choices')
labels=['No module','Mean Teacher\n(C2)','Metadata FiLM\n(C3)','Colour-map stem\n(C4)'];xs=[.25,.46,.67,.88]
for x,l in zip(xs,labels):ax.text(x,.94,l,ha='center',fontsize=8)
for y,l,color in [(.68,'Baseline',BLUE),(.39,'SSL',ORANGE)]:
 ax.text(.0,y,l,va='center',fontsize=9,fontweight='bold',color=color)
 for x in xs:
  ax.add_patch(Rectangle((x-.077,y-.10),.154,.20,transform=ax.transAxes,facecolor=color,alpha=.10,edgecolor=color))
  ax.text(x,y,'10 seeds',ha='center',va='center',fontsize=8,color=color)
ax.text(.0,.08,'Additional checks',fontweight='bold',fontsize=8.5);ax.text(.25,.08,'Wider SSL frame gap: 5 seeds',fontsize=8);ax.text(.65,.08,'HIT-UAV baseline: 10 seeds',fontsize=8)
save(fig,'Fig1_study_design');(A/'example_images.json').write_text(json.dumps(examples))

# Figure 2: paired seed data and exhaustive subset sensitivity.
b=arr('Baseline');s=arr('SSL');d=s-b
fig,axs=plt.subplots(2,2,figsize=(7.2,6.1));fig.subplots_adjust(hspace=.57,wspace=.40)
ax=axs[0,0]
for i in range(10):ax.plot([0,1],[b[i],s[i]],color='#A9ADB1',lw=.8,zorder=1)
ax.scatter(np.zeros(10),b,s=25,c=BLUE,zorder=2);ax.scatter(np.ones(10),s,s=25,c=ORANGE,zorder=2)
for x,v,col in [(0,b,BLUE),(1,s,ORANGE)]:m,h=ci(v);ax.errorbar(x+.14,m,yerr=h,fmt='s',color=col,capsize=3,ms=5)
ax.set_xticks([0,1],['Baseline','SSL']);ax.set_xlim(-.3,1.45);ax.set_ylabel('Test mAP50');panel(ax,'a','Ten matched training seeds')
ax=axs[0,1];colors=[ORANGE if x>0 else BLUE for x in d];ax.bar(np.arange(42,52),d,color=colors,width=.64);zero(ax);ax.set_xticks(range(42,52),range(42,52),rotation=45);ax.set_ylabel('SSL − baseline mAP50');ax.set_xlabel('Downstream seed');panel(ax,'b','The direction varies across seeds')
sub=np.array([d[list(ix)].mean() for ix in itertools.combinations(range(10),5)])
ax=axs[1,0];ax.hist(sub,bins=np.linspace(-.045,.06,23),color=BLUE,alpha=.8,edgecolor='white');zero(ax,True);ax.axvline(d.mean(),color=ORANGE,lw=1.5,label='Ten-seed mean');ax.set_xlabel('Five-seed mean difference');ax.set_ylabel('Number of subsets');ax.set_ylim(0,33);ax.text(.04,.96,'161 / 252\npositive subsets',transform=ax.transAxes,va='top',fontsize=8);ax.legend(frameon=False,fontsize=8,loc='upper right');panel(ax,'c','Five-seed estimates span both signs')
ax=axs[1,1];ks=np.arange(2,11);sets=[np.array([d[list(ix)].mean() for ix in itertools.combinations(range(10),int(k))]) for k in ks]
ax.fill_between(ks,[x.min() for x in sets],[x.max() for x in sets],color=BLUE,alpha=.14,label='Full subset range');ax.fill_between(ks,[np.quantile(x,.025) for x in sets],[np.quantile(x,.975) for x in sets],color=BLUE,alpha=.26,label='Central 95% of subsets');ax.plot(ks,[d[:k].mean() for k in ks],'o-',color=ORANGE,ms=3,label='Seeds in numerical order');zero(ax);ax.set_xlabel('Number of paired seeds in subset');ax.set_ylabel('Mean mAP50 difference');ax.set_xticks([2,4,6,8,10]);ax.legend(frameon=False,fontsize=7,loc='lower right');panel(ax,'d','Sensitivity to subset size')
save(fig,'Fig2_seed_sensitivity')

# Figure 3: class-specific paired data and cancellation.
fig,axs=plt.subplots(2,2,figsize=(7.2,6.0));fig.subplots_adjust(hspace=.56,wspace=.42)
for ax,k,letter,col in zip(axs[0],['Oil','Water'],['a','b'],[ORANGE,BLUE]):
 bb=arr('Baseline',k+'_AP50');ss=arr('SSL',k+'_AP50')
 for i in range(10):ax.plot([0,1],[bb[i],ss[i]],color='#B7BBBD',lw=.8)
 ax.scatter(np.zeros(10),bb,color=BLUE,s=23,zorder=3);ax.scatter(np.ones(10),ss,color=ORANGE,s=23,zorder=3)
 for x,v,c in [(0,bb,BLUE),(1,ss,ORANGE)]:m,h=ci(v);ax.errorbar(x+.14,m,yerr=h,fmt='s',color=c,capsize=3,ms=5)
 ax.set_xticks([0,1],['Baseline','SSL']);ax.set_xlim(-.3,1.45);ax.set_ylabel(f'{k} AP50');ax.set_ylim(.25,.85);panel(ax,letter,f'{k} class across matched runs')
ax=axs[1,0]
for y,(name,st,col) in enumerate([('Oil',S['classes']['Oil'],ORANGE),('Water',S['classes']['Water'],BLUE),('Aggregate',S['comparisons']['SSL'],GRAY)]):
 ax.errorbar(st['mean'],y,xerr=[[st['mean']-st['lo']],[st['hi']-st['mean']]],fmt='o',color=col,capsize=3,ms=5);ax.text(.105,y,f"{st['positive']}/10 higher",fontsize=7,va='center')
ax.set_yticks(range(3),['Oil','Water','mAP50']);ax.invert_yaxis();ax.set_xlim(-.065,.165);ax.set_xlabel('SSL − baseline difference (95% CI)');zero(ax,True);panel(ax,'c','Opposing means partly cancel')
ax=axs[1,1];do=arr('SSL','Oil_AP50')-arr('Baseline','Oil_AP50');dw=arr('SSL','Water_AP50')-arr('Baseline','Water_AP50')
zero(ax);zero(ax,True);xx=np.linspace(-.22,.17,100);ax.plot(xx,-xx,'--',color=GRAY,lw=.8,label='Unchanged mAP50');ax.scatter(do,dw,color=PURPLE,s=29)
for i,(x,y) in enumerate(zip(do,dw)):ax.annotate(str(42+i),(x,y),xytext=(3,4),textcoords='offset points',fontsize=7)
ax.scatter(do.mean(),dw.mean(),marker='D',color='black',s=38,label='Mean');ax.set_xlim(-.20,.23);ax.set_ylim(-.14,.12);ax.set_xlabel('Change in Oil AP50');ax.set_ylabel('Change in Water AP50');ax.legend(frameon=False,fontsize=7,loc='upper left');panel(ax,'d','Per-seed class trade-offs')
save(fig,'Fig3_class_effects')

# Figure 4: benchmark export distributions and variability.
cats=['Bicycle','Car','DontCare','OtherVehicle','Person'];vals=[np.array([r['per_class_ap50'][k] for r in D['HIT-UAV']]) for k in cats];vals.append(arr('HIT-UAV'));labs=cats+['mAP50']
fig=plt.figure(figsize=(7.2,5.9));gs=fig.add_gridspec(2,2,height_ratios=[1,1.2],hspace=.60,wspace=.60)
ax=fig.add_subplot(gs[0,:]);cols=[BLUE,BLUE,GRAY,ORANGE,BLUE,PURPLE]
for i,(v,col) in enumerate(zip(vals,cols)):
 ax.scatter(i+np.linspace(-.12,.12,10),v,s=17,color=col,alpha=.65);m,h=ci(v);ax.errorbar(i+.22,m,yerr=h,fmt='D',ms=4,capsize=3,color=col)
ax.set_xticks(range(6),labs);ax.set_ylabel('Test AP50 or mAP50');ax.set_ylim(.35,1.01);panel(ax,'a','Ten baseline runs on the HIT-UAV export')
ax=fig.add_subplot(gs[1,0]);sds=[v.std(ddof=1) for v in vals];ax.hlines(range(6),0,sds,color=cols);ax.scatter(sds,range(6),c=cols,s=28);ax.set_yticks(range(6),labs);ax.invert_yaxis();ax.set_xlabel('Across-seed sample standard deviation');ax.set_xlim(0,.09);panel(ax,'b','OtherVehicle varies most')
for i,v in enumerate(sds):ax.text(v+.002,i,f'{v:.3f}',va='center',fontsize=7)
ax=fig.add_subplot(gs[1,1]);mat=np.array([v-v.mean() for v in vals]);im=ax.imshow(mat,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(0,-.16,.16));ax.set_yticks(range(6),labs);ax.set_xticks(range(10),range(42,52),rotation=45);ax.set_xlabel('Downstream seed');panel(ax,'c','Centred scores across seeds');c=fig.colorbar(im,ax=ax,pad=.04,fraction=.045);c.ax.tick_params(labelsize=7);c.set_label('AP deviation',fontsize=8)
save(fig,'Fig4_hituav_variability')

# Figure 5: full configuration evidence and interaction tests.
keys=['Baseline','SSL','Baseline+C2','SSL+C2','Baseline+C3','SSL+C3','Baseline+C4','SSL+C4']
fig=plt.figure(figsize=(7.2,6.0));gs=fig.add_gridspec(2,2,height_ratios=[1,1.15],hspace=.66,wspace=.57)
ax=fig.add_subplot(gs[0,:])
for i,k in enumerate(keys):
 v=arr(k);col=ORANGE if k.startswith('SSL') else BLUE;ax.scatter(i+np.linspace(-.13,.13,len(v)),v,color=col,s=18,alpha=.62);m,h=ci(v);ax.errorbar(i+.2,m,yerr=h,color=col,fmt='s',ms=4,capsize=3)
ax.set_xticks(range(8),['Baseline','SSL','Base+C2','SSL+C2','Base+C3','SSL+C3','Base+C4','SSL+C4']);ax.tick_params(axis='x',labelsize=7);ax.set_ylim(.36,.74);ax.set_ylabel('Test mAP50');panel(ax,'a','Run-level scores for all eight configurations')
ax=fig.add_subplot(gs[1,0]);mods=['C2','C3','C4']
for i,k in enumerate(mods):
 for off,prefix,col,marker in [(-.12,'Baseline+',BLUE,'o'),(.12,'SSL+',ORANGE,'s')]:
  st=S['comparisons'][prefix+k];ax.errorbar(st['mean'],i+off,xerr=[[st['mean']-st['lo']],[st['hi']-st['mean']]],fmt=marker,color=col,ms=4,capsize=2,label=prefix[:-1] if i==0 else None)
ax.set_yticks(range(3),['C2 Mean Teacher','C3 Metadata FiLM','C4 Colour-map']);ax.invert_yaxis();zero(ax,True);ax.set_xlabel('Module effect (95% CI)');ax.legend(frameon=False,fontsize=7,loc='lower left');panel(ax,'b','Paired effects within each initialisation')
ax=fig.add_subplot(gs[1,1])
for i,k in enumerate(mods):
 st=S['interactions'][k];ax.errorbar(st['mean'],i,xerr=[[st['mean']-st['lo']],[st['hi']-st['mean']]],fmt='D',color=PURPLE,ms=4,capsize=3);ax.text(.055,i,f"P = {st['p']:.3f}",fontsize=7,va='center')
zero(ax,True);ax.set_yticks(range(3),mods);ax.invert_yaxis();ax.set_xlim(-.125,.11);ax.set_xlabel('Difference in module effects (95% CI)');panel(ax,'c','Interaction with SSL initialisation')
save(fig,'Fig5_interventions')

# Figure 6: exact prospective sensitivity, not plug-in observed power.
fig,axs=plt.subplots(1,2,figsize=(7.2,3.15));fig.subplots_adjust(wspace=.42)
power=TTestPower();ax=axs[0];effects=np.linspace(.10,1.0,130);ns=[np.ceil(power.solve_power(effect_size=float(e),alpha=.05,power=.8)) for e in effects];ax.plot(effects,ns,color=BLUE)
for e,n in [(.2,199),(.4,52),(.8,15)]:ax.scatter(e,n,color=BLUE,s=28);ax.annotate(f'{n} pairs',(e,n),xytext=(7,6),textcoords='offset points',fontsize=8)
ax.scatter(S['comparisons']['SSL']['dz'],470,marker='D',s=24,color=GRAY);ax.annotate('Pilot-based illustration',(S['comparisons']['SSL']['dz'],470),xytext=(.30,580),textcoords='data',arrowprops={'arrowstyle':'-','lw':.7,'color':GRAY},fontsize=7.5,color=GRAY)
ax.set_yscale('log');ax.set_ylim(7,1000);ax.set_yticks([10,20,50,100,200,500,1000],['10','20','50','100','200','500','1,000']);ax.set_xlabel('Assumed paired effect size |d$_z$|');ax.set_ylabel('Pairs for 80% power');panel(ax,'a','Effect size and seed budget')
ax=axs[1];nn=np.arange(3,101)
for e,col in [(.2,BLUE),(.4,GREEN),(.8,ORANGE)]:ax.plot(nn,power.power(effect_size=e,nobs=nn,alpha=.05),color=col,label=f'|d$_z$| = {e:.1f}')
ax.axhline(.8,color=GRAY,linestyle='--',lw=.8);ax.axvline(10,color=GRAY,linestyle=':',lw=.8);ax.text(12,.04,'10 seeds',fontsize=7,color=GRAY);ax.set_xlabel('Number of paired seeds');ax.set_ylabel('Power under the stated assumptions');ax.set_ylim(0,1.02);ax.set_xlim(3,100);ax.legend(frameon=False,fontsize=8,loc='lower right');panel(ax,'b','Power at fixed seed budgets')
save(fig,'Fig6_power_sensitivity')

# Supplementary diagnostic: complete pretraining curve and gap ablation.
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0));fig.subplots_adjust(wspace=.4)
hist=json.loads((ROOT/'result_data_paper4/ssl_pretrain/ssl_pretrain_summary.json').read_text())['history'];axs[0].plot([r['epoch']+1 for r in hist],[r['loss'] for r in hist],color=BLUE);axs[0].set_xlabel('Pretraining epoch');axs[0].set_ylabel('Dense contrastive loss');panel(axs[0],'a','Pretext optimisation')
g=arr('SSL gap6');s5=s[:5]
for i in range(5):axs[1].plot([0,1],[s5[i],g[i]],color='#B7BBBD',lw=.8)
axs[1].scatter(np.zeros(5),s5,color=ORANGE,s=24);axs[1].scatter(np.ones(5),g,color=PURPLE,s=24);axs[1].set_xticks([0,1],['Gap 1–2','Gap 1–6']);axs[1].set_ylabel('Downstream mAP50');axs[1].set_xlim(-.3,1.3);panel(axs[1],'b','Wider temporal gap in five matched runs');save(fig,'FigS1_ssl_diagnostics')
print('Saved six main figures and one supplementary figure in PNG, PDF and SVG.')

# Distribution diagnostics for all 13 paired contrasts.
Q=json.loads((A/'diagnostics.json').read_text())
fig,axs=plt.subplots(4,4,figsize=(8.4,8.3));fig.subplots_adjust(hspace=.70,wspace=.55)
for i,(name,r) in enumerate(Q.items()):
 ax=axs.flat[i];x=np.array(r['differences']);(osm,osr),(slope,intercept,corr)=stats.probplot(x,dist='norm')
 ax.scatter(osm,osr,s=18,color=ORANGE if name=='Baseline+C2' else BLUE)
 ax.plot(osm,slope*np.array(osm)+intercept,color=GRAY,lw=1.5)
 ax.set_title(name,loc='left',fontsize=9,fontweight='bold',pad=6)
 ax.text(.04,.96,f"n = {r['n']}\nW = {r['shapiro_w']:.3f}\nP = {r['shapiro_p']:.3f}",transform=ax.transAxes,fontsize=8,va='top')
 ax.tick_params(labelsize=8);ax.set_xlabel('Normal quantile',fontsize=8);ax.set_ylabel('Paired difference',fontsize=8)
for ax in list(axs.flat)[len(Q):]:ax.axis('off')
save(fig,'FigS2_distribution_diagnostics')
print('All v3 figures and distribution diagnostics exported.')
