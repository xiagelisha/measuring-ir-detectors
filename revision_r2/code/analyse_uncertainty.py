"""Crossed bootstrap of fixed trained detectors and shared test images.

AP is recomputed from pooled detections, never averaged over image-level AP.
Cluster resampling draws two filename groups within each of the two classes.
These are descriptive intervals over the four observed groups, not site CIs.
"""
from pathlib import Path
import numpy as np
import json, csv, itertools, argparse
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'doc/20260628/maijorRevise20261014/revision_r2_20261014'
A=OUT/'analysis'

def compute_ap(tp, n):
    if len(tp)==0: return 0.
    tpc=np.cumsum(tp,dtype=np.float64)
    recall=tpc/n; precision=tpc/np.arange(1,len(tp)+1)
    mrec=np.r_[0.,recall,recall[-1],1.]
    mpre=np.r_[1.,precision,0.,0.]
    mpre=np.maximum.accumulate(mpre[::-1])[::-1]
    return float(np.trapezoid(np.interp(np.linspace(0,1,101),mrec,mpre),np.linspace(0,1,101)))

class Run:
    def __init__(self,path,names):
        z=np.load(path)
        assert list(z['names'])==list(names), 'Image order differs between checkpoints'
        self.z=z; self.pred=[]; self.gt=[]
        pimg=np.repeat(np.arange(len(names)),z['pred_counts'])
        gimg=np.repeat(np.arange(len(names)),z['target_counts'])
        # Use the validator's global confidence ordering for deterministic ties.
        order=np.argsort(-z['conf'])
        for c in range(2):
            idx=order[z['pred_cls'][order]==c]
            self.pred.append((pimg[idx],z['tp'][idx,0]))
            self.gt.append(np.bincount(gimg[z['target_cls']==c],minlength=len(names)))
    def ap(self,weights):
        result=[]
        for c in range(2):
            n=int(weights@self.gt[c]); im,tp=self.pred[c]
            result.append(compute_ap(np.repeat(tp,weights[im]),n) if n else np.nan)
        return np.array([np.mean(result),result[0],result[1]])

def ci(x):return np.quantile(x,[.025,.975],axis=0).T.tolist()
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    global OUT,A
    p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=2000)
    p.add_argument('--bundle',type=Path,help='Extracted Source_Data_R2 folder for standalone reanalysis')
    args=p.parse_args()
    if args.bundle:
        OUT=args.bundle.resolve();A=OUT/'analysis';A.mkdir(exist_ok=True)
    A.mkdir(parents=True,exist_ok=True)
    B=args.replicates;rng=np.random.default_rng(20261014)
    names=np.load(OUT/'evaluation/Baseline_42.npz')['names'];N=len(names)
    meta_path=OUT/'metadata.csv' if args.bundle else ROOT/'data/metadata.csv'
    meta={r['image_name']:r for r in csv.DictReader(meta_path.open(encoding='utf-8-sig'))}
    groups=np.array([meta[Path(n).stem]['session_id'] for n in names]);unique=sorted(set(groups))
    group_class={g:int(meta[Path(names[np.where(groups==g)[0][0]]).stem]['has_oil']) for g in unique}
    weights=rng.multinomial(N,np.full(N,1/N),size=B).astype(np.int16)
    # All nine distinct class-stratified cluster-count vectors, with exact probabilities.
    cluster_weights=[];cluster_prob=[];cluster_counts=[]
    for k0,k1 in itertools.product(range(3),repeat=2):
        counts={}
        for c,k in [(0,k0),(1,k1)]:
            gs=[g for g in unique if group_class[g]==c];assert len(gs)==2
            counts.update({gs[0]:k,gs[1]:2-k})
        cluster_weights.append(np.array([counts[g] for g in groups],dtype=np.int16))
        cluster_prob.append([.25,.5,.25][k0]*[.25,.5,.25][k1]);cluster_counts.append(counts)
    scores=np.empty((20,B,3));clusters=np.empty((20,9,3));points=[];rows=[];gr=[]
    for j,(kind,seed) in enumerate(itertools.product(['Baseline','SSL'],range(42,52))):
        run=Run(OUT/'evaluation'/f'{kind}_{seed}.npz',names)
        point=run.ap(np.ones(N,dtype=int));points.append(point)
        prior=json.loads((OUT/'evaluation'/f'{kind}_{seed}.json').read_text())['actual']
        assert np.max(np.abs(point-[prior['mAP50'],prior['Water_AP50'],prior['Oil_AP50']]))<1e-10
        cache=A/f'bootstrap_{kind}_{seed}_{B}.npy'
        if cache.exists(): scores[j]=np.load(cache)
        else:
            for b,w in enumerate(weights):scores[j,b]=run.ap(w)
            np.save(cache,scores[j])
        clusters[j]=[run.ap(w) for w in cluster_weights]
        for g in unique:
            aps=run.ap((groups==g).astype(int));c=group_class[g]
            gr.append({'configuration':kind,'seed':seed,'group':g,'class':'Oil' if c else 'Water','images':int(sum(groups==g)),'AP50':float(aps[c+1])})
        intervals=ci(scores[j]);row={'configuration':kind,'seed':seed}
        for m,val,interval in zip(['mAP50','Water_AP50','Oil_AP50'],point,intervals):
            row.update({m:val,m+'_image_low':interval[0],m+'_image_high':interval[1]})
        rows.append(row);print('BOOTSTRAP',kind,seed,flush=True)
    points=np.array(points)
    # Crossed resampling: one shared image sample and seed sample per replicate.
    s0=rng.integers(0,10,size=(B,10));s1=rng.integers(0,10,size=(B,10));bidx=np.arange(B)[:,None]
    independent=(scores[10+s1,bidx].mean(axis=1)-scores[s0,bidx].mean(axis=1))
    matched=(scores[10+s0,bidx]-scores[s0,bidx]).mean(axis=1)
    testonly=scores[10:].mean(axis=0)-scores[:10].mean(axis=0)
    K=20000;ci_idx=rng.choice(9,K,p=cluster_prob);a=rng.integers(0,10,(K,10));b=rng.integers(0,10,(K,10))
    cluster_combined=clusters[10+b,ci_idx[:,None]].mean(axis=1)-clusters[a,ci_idx[:,None]].mean(axis=1)
    cluster_matched=(clusters[10+a,ci_idx[:,None]]-clusters[a,ci_idx[:,None]]).mean(axis=1)
    cluster_testonly=(clusters[10:].mean(axis=0)-clusters[:10].mean(axis=0))[ci_idx]
    for j,row in enumerate(rows):
        for m,interval in zip(['mAP50','Water_AP50','Oil_AP50'],ci(clusters[j,ci_idx])):
            row[m+'_cluster_low'],row[m+'_cluster_high']=interval
    summary={'bootstrap_replicates':B,'bootstrap_seed':20261014,'cluster_seed_replicates':K,'group_classes':group_class,
        'point_difference':(points[10:].mean(axis=0)-points[:10].mean(axis=0)).tolist(),
        'metric_order':['mAP50','Water_AP50','Oil_AP50'],
        'image_only':ci(testonly),'seed_and_image_independent':ci(independent),'seed_and_image_matched':ci(matched),
        'cluster_only':ci(cluster_testonly),'seed_and_cluster_independent':ci(cluster_combined),'seed_and_cluster_matched':ci(cluster_matched),
        'cluster_vectors':cluster_counts,'cluster_probabilities':cluster_prob,
        'notes':['Percentile intervals. Shared test sample across all models in each replicate.',
                 'Independent seed resampling is the primary sensitivity; nominal matching is also retained.',
                 'Cluster bootstrap resamples two groups per class; only four observed groups, nine distinct count vectors.',
                 'Nonrepresented class AP is undefined for per-group evaluation.']}
    (A/'uncertainty_summary.json').write_text(json.dumps(summary,indent=2))
    write_csv(A/'per_run_test_uncertainty.csv',rows);write_csv(A/'per_group_ap.csv',gr)
    np.savez_compressed(A/'bootstrap_distributions.npz',per_run=scores,clusters=clusters,
        image_only=testonly,seed_image=independent,seed_image_matched=matched,
        cluster_only=cluster_testonly,seed_cluster=cluster_combined,seed_cluster_matched=cluster_matched)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
