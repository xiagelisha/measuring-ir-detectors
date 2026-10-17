"""Re-evaluate saved primary checkpoints; retain exact validator AP statistics.

Run with the frozen oilleak environment. No training or checkpoint changes.
"""
from pathlib import Path
import os, json, argparse, hashlib
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'doc/20260628/maijorRevise20261014/revision_r2_20261014'
(OUT / 'evaluation').mkdir(parents=True, exist_ok=True)
config_dir = ROOT / 'tmp/r2_yolo_config'
config_dir.mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

class CaptureValidator(DetectionValidator):
    destination = None
    def init_metrics(self, model):
        super().init_metrics(model)
        self.records = []
        original = self.metrics.update_stats
        def record(stat):
            self.records.append({k: v.copy() if isinstance(v, np.ndarray) else v for k,v in stat.items()})
            original(stat)
        self.metrics.update_stats = record

    def get_stats(self):
        rec = self.records
        np.savez_compressed(self.destination,
            names=np.array([r['im_name'] for r in rec]),
            pred_counts=np.array([len(r['conf']) for r in rec]),
            target_counts=np.array([len(r['target_cls']) for r in rec]),
            **{k: np.concatenate([r[k] for r in rec]) for k in ('tp','conf','pred_cls','target_cls')})
        return super().get_stats()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=20)
    args=parser.parse_args()
    summaries=[]
    for kind in ['Baseline','SSL']:
        for seed in range(42,52):
            if len(summaries)>=args.limit: break
            if kind=='Baseline':
                name=f'ms1_d4a_baseline_seed{seed}' if seed<47 else f'baseline_ext_n10_seed{seed}'
            else:
                name=f'ms2_d8b_ssl_only_diag_seed{seed}' if seed<47 else f'ssl_only_ext_n10_seed{seed}'
            run=ROOT/'results_v2/methods/stc_yolo'/name
            dest=OUT/'evaluation'/f'{kind}_{seed}.npz'
            meta_path=dest.with_suffix('.json')
            if meta_path.exists() and dest.exists():
                summaries.append(json.loads(meta_path.read_text()))
                continue
            prior=json.loads((run/'ultralytics_test_metrics.json').read_text())
            CaptureValidator.destination=dest
            checkpoint=run/'train/weights/best.pt'
            print(f'BEGIN {kind} {seed}',flush=True)
            model=YOLO(str(checkpoint))
            result=model.val(validator=CaptureValidator,
                data=str(ROOT/'results_v2/orchestrator/runs/_data_yaml/data.yaml'),
                split='test',imgsz=1056,augment=True,conf=.001,iou=.6,
                project=str(OUT/'evaluation/validator'),name=f'{kind}_{seed}',
                exist_ok=True,plots=False,verbose=False,workers=0)
            actual={'mAP50':float(result.box.map50),'Water_AP50':float(result.box.ap50[0]),'Oil_AP50':float(result.box.ap50[1])}
            errors={k:actual[k]-prior[k] for k in actual}
            meta={'configuration':kind,'seed':seed,'run_name':name,'checkpoint':str(checkpoint.relative_to(ROOT)),
                  'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'ultralytics':ultralytics.__version__,'torch':torch.__version__,
                  'actual':actual,'submitted':{k:prior[k] for k in actual},'difference':errors,
                  'inference':{'imgsz':1056,'augment':True,'conf':.001,'iou':.6}}
            meta_path.write_text(json.dumps(meta,indent=2))
            summaries.append(meta)
            print('COMPLETE',kind,seed,json.dumps(errors),flush=True)
            if max(map(abs,errors.values()))>1e-5:
                raise RuntimeError('Metric reproduction discrepancy; investigate before continuing')
    (OUT/'evaluation/reproduction.json').write_text(json.dumps(summaries,indent=2))

if __name__=='__main__': main()
