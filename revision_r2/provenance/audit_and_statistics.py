from pathlib import Path
import os,json,csv,collections
ROOT=Path(__file__).resolve().parents[2]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'tmp/r2_yolo_config')
import numpy as np
from scipy import stats
import torch
from ultralytics import YOLO
OUT=ROOT/'doc/20260628/maijorRevise20261014/revision_r2_20261014'
A=OUT/'analysis';D=json.loads((A/'submitted_source/source_data/data.json').read_text())
def arr(k,m='mAP50'):return np.array([r[m] for r in D[k]])
def contrast(k,ref):
    x=arr(k);y=arr(ref)[:len(x)];d=x-y;n=len(x)
    v=x.var(ddof=1)/n+y.var(ddof=1)/n;df=v*v/((x.var(ddof=1)/n)**2/(n-1)+(y.var(ddof=1)/n)**2/(n-1))
    h=stats.t.ppf(.975,df)*np.sqrt(v);hp=stats.t.ppf(.975,n-1)*d.std(ddof=1)/np.sqrt(n)
    return {'configuration':k,'reference':ref,'n':n,'mean':float(x.mean()),'sd':float(x.std(ddof=1)),
            'delta':float(d.mean()),'paired_low':float(d.mean()-hp),'paired_high':float(d.mean()+hp),
            'correlation':float(np.corrcoef(x,y)[0,1]),'paired_variance_ratio':float(d.var(ddof=1)/(x.var(ddof=1)+y.var(ddof=1))),
            'welch_low':float(d.mean()-h),'welch_high':float(d.mean()+h),'welch_df':float(df),'welch_p':float(stats.ttest_ind(x,y,equal_var=False).pvalue)}
contrasts=[contrast('SSL','Baseline')]+[contrast(k,r) for r in ['Baseline','SSL'] for k in [r+'+C2',r+'+C3',r+'+C4']]+[contrast('SSL gap6','SSL')]
classes=['Bicycle','Car','DontCare','OtherVehicle','Person'];hit=np.array([[r['per_class_ap50'][c] for c in classes] for r in D['HIT-UAV']]);cov=np.cov(hit,rowvar=False)
result={'contrasts':contrasts,'oil_water_delta_correlation':float(np.corrcoef(arr('SSL','Oil_AP50')-arr('Baseline','Oil_AP50'),arr('SSL','Water_AP50')-arr('Baseline','Water_AP50'))[0,1]),
 'hituav_variance':{'class_order':classes,'covariance_matrix':cov.tolist(),'sum_class_variances':float(np.trace(cov)),
 'variance_component':float(np.trace(cov)/25),'covariance_component':float((cov.sum()-np.trace(cov))/25),'observed_aggregate_variance':float(arr('HIT-UAV').var(ddof=1)),
 'othervehicle_variance_share':float(cov[3,3]/np.trace(cov)),'sd_if_zero_covariance':float(np.sqrt(np.trace(cov)/25))}}
meta=list(csv.DictReader((ROOT/'data/metadata.csv').open(encoding='utf-8-sig')));rows=[]
for split in ['train','val','test']:
    stems=set((ROOT/f'data/splits/main_holdout/{split}.txt').read_text().splitlines())
    mm=[r for r in meta if r['image_name'] in stems]
    for g in sorted(set(r['session_id'] for r in mm)):
        rr=[r for r in mm if r['session_id']==g]; counts=collections.Counter()
        for r in rr:
            lines=(ROOT/r['label_path']).read_text().splitlines();labels=[int(x.split()[0]) for x in lines if x.strip()]
            assert set(labels)==({1} if int(r['has_oil']) else {0})
            counts.update(labels)
        rows.append({'split':split,'group':g,'images':len(rr),'Oil_images':sum(int(r['has_oil']) for r in rr),'Water_images':sum(int(r['has_water']) for r in rr),'Oil_boxes':counts[1],'Water_boxes':counts[0]})
with (A/'group_allocation.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result['condition_class_counts']={c:dict(collections.Counter('Oil' if int(r['has_oil']) else 'Water' for r in meta if r['condition']==c)) for c in sorted(set(r['condition'] for r in meta))}
model=YOLO(str(ROOT/'yolo12s.pt')).model
base=model.state_dict(); param_names=set(dict(model.named_parameters()))
ck=ROOT/D['SSL'][0]['stc_config']['ssl_pretrained_backbone']
ssl=torch.load(ck,map_location='cpu',weights_only=False)['state_dict']
change={'parameters':collections.Counter(),'buffers':collections.Counter()};max_delta={}
for k,v in ssl.items():
    if k not in base or v.shape!=base[k].shape:continue
    if not torch.equal(v,base[k]):
        category='parameters' if k in param_names else 'buffers'; layer=k.split('.')[1]
        change[category][layer]+=1
        max_delta[k]=float((v.to(torch.float64)-base[k].to(torch.float64)).abs().max())
result['ssl_checkpoint_audit']={'checkpoint':str(ck.relative_to(ROOT)),'changed_tensors_by_layer':change,'max_absolute_changes':max_delta}
(A/'additional_statistics.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='ssl_checkpoint_audit'},indent=2))
print('SSL CHANGES',change)
