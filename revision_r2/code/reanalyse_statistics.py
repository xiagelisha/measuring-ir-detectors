"""Recompute run-level revision statistics and verify the supplied summaries."""
from pathlib import Path
import argparse, json
import numpy as np
from scipy import stats

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle',type=Path,required=True)
args=parser.parse_args()
bundle=args.bundle.resolve()
D=json.loads((bundle/'original_source_data/data.json').read_text())
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

expected=json.loads((bundle/'analysis/additional_statistics.json').read_text())
checks=0
def verify(actual,reference,path):
    global checks
    if isinstance(actual,dict):
        for key,value in actual.items():verify(value,reference[key],path+'.'+key)
    elif isinstance(actual,list):
        assert len(actual)==len(reference),path
        for i,(value,ref) in enumerate(zip(actual,reference)):verify(value,ref,path+f'[{i}]')
    elif isinstance(actual,(int,float)):
        assert np.isclose(actual,reference,rtol=1e-8,atol=1e-10),f'{path}: {actual} != {reference}'
        checks+=1
    else:
        assert actual==reference,path
verify(result,expected,'statistics')
(bundle/'analysis/recomputed_statistics.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'PASS: {checks} numerical statistics agree with the supplied revision results.')
