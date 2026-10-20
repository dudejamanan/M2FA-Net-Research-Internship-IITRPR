"""Reproducible, inference-only audit; never imports the training script's side effects."""
from pathlib import Path
import argparse, ast, csv, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50 as make_deeplab

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'Dataset'/'WeedyRice-RGBMS-DB'
SOURCE = ROOT/'Multi Level Fusion'/'MultiLevelFusion.py.py'
OUT = ROOT/'reviewer9_audit'/'checkpoint_analysis'

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_model(checkpoint):
    tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('CBAM','MultiLevelFusionDeepLab')]
    # Pretrained initialization is unnecessary: strict checkpoint loading replaces all tensors.
    class Weights:
        COCO_WITH_VOC_LABELS_V1=None
    def offline_deeplab(**kwargs):
        return make_deeplab(weights=None,weights_backbone=None)
    namespace=dict(torch=torch,nn=nn,F=F,DeepLabV3_ResNet50_Weights=Weights,deeplabv3_resnet50=offline_deeplab)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),namespace)
    model=namespace['MultiLevelFusionDeepLab']()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    for key in ('state_dict','model_state_dict'):
        if key in state: state=state[key]; break
    loaded=model.load_state_dict(state,strict=True)
    model.eval()
    print(f'Strict checkpoint load: {loaded}; tensors={len(state)}',flush=True)
    return model,len(state)

def ms_input(base):
    bands=[]
    for suffix in ('_G.TIF','_R.TIF','_RE.TIF','_NIR.TIF'):
        path=DATA/'Multispectral'/(base+suffix)
        a=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if a is None: raise FileNotFoundError(path)
        a=cv2.resize(a,(512,512))
        if a.ndim==3: a=a[:,:,0]
        bands.append(a.astype(np.float32)/65535.0)
    G,R,RE,NIR=bands
    ndvi=((NIR-R)/(NIR+R+1e-6)+1)/2
    ndre=((NIR-RE)/(NIR+RE+1e-6)+1)/2
    return np.stack([G,R,RE,NIR,ndvi,ndre],axis=-1)

def sample_stats(base):
    a=ms_input(base)
    return a.mean(axis=(0,1)),a.std(axis=(0,1))

def reconstruct():
    names=(DATA/'train_list.txt').read_text().splitlines()
    bases=[Path(n.strip()).stem for n in names]
    cv2.setNumThreads(1)
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as pool:
        stats=list(pool.map(sample_stats,bases))
    # Match the local function's mean-of-image-means and mean-of-image-stds recipe.
    mean=np.asarray([s[0] for s in stats],dtype=np.float64).mean(axis=0)
    std=np.asarray([s[1] for s in stats],dtype=np.float64).mean(axis=0)
    record=dict(mean=mean.tolist(),std=std.tolist(),source='reconstructed from supplied training split',
                provisional=True,images=len(bases),recipe='mean of per-image channel means; mean of per-image population standard deviations; matches local compute_ms_statistics',
                split_sha256=sha256(DATA/'train_list.txt'))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'reconstructed_normalization.json').write_text(json.dumps(record,indent=2))
    print(f'Reconstructed training normalization in {time.monotonic()-started:.1f}s: {record}',flush=True)
    return record

def load_stats(path):
    if path:
        path=Path(path).resolve()
        if path.suffix.lower()=='.json': record=json.loads(path.read_text())
        else:
            obj=np.load(path,allow_pickle=True).item()
            record={'mean':np.asarray(obj['mean']).tolist(),'std':np.asarray(obj['std']).tolist()}
        record.update(source=str(path),sha256=sha256(path),provisional=record.get('provisional',False))
    else:
        cached=OUT/'reconstructed_normalization.json'
        record=json.loads(cached.read_text()) if cached.exists() else reconstruct()
    mean=np.asarray(record['mean'],np.float32).reshape(1,1,6)
    std=np.asarray(record['std'],np.float32).reshape(1,1,6)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('Invalid normalization values')
    return record,mean,std

def prepare(filename,mean,std):
    base=Path(filename).stem
    im=cv2.imread(str(DATA/'RGB'/filename))
    if im is None: raise FileNotFoundError(filename)
    rgb=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(512,512)).astype(np.float32)/255
    x=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    ms=(ms_input(base)-mean)/std
    native=cv2.imread(str(DATA/'Masks'/(base+'.png')),0)
    # Preserve the original evaluation rule; also score against nearest-neighbor GT for transparency.
    gt=cv2.resize(native,(512,512))>0
    nearest=cv2.resize(native,(512,512),interpolation=cv2.INTER_NEAREST)>0
    return torch.from_numpy(x.transpose(2,0,1).copy()).unsqueeze(0),torch.from_numpy(ms.transpose(2,0,1).copy()).unsqueeze(0),gt,nearest

def scores(pred,gt):
    tp=int((pred&gt).sum()); fp=int((pred&~gt).sum()); fn=int((~pred&gt).sum())
    return dict(tp=tp,fp=fp,fn=fn,iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 1.)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stats',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--threads',type=int,default=6)
    args=parser.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'best_model_MultiLevelFusionDeepLab.pth'
    started=time.monotonic()
    model,tensors=load_model(checkpoint)
    print(f'Checkpoint ready in {time.monotonic()-started:.1f}s',flush=True)
    if args.prepare_only:
        with torch.inference_mode():
            t=time.monotonic(); y=model(torch.zeros(1,3,512,512),torch.zeros(1,6,512,512))
        print(f'CPU benchmark: output={list(y.shape)}, finite={bool(torch.isfinite(y).all())}, seconds={time.monotonic()-t:.2f}',flush=True)
        if not (OUT/'reconstructed_normalization.json').exists(): reconstruct()
        return
    stats_path=args.stats
    if stats_path is None and (ROOT/'ms_mean_std.npy').exists():
        stats_path=ROOT/'ms_mean_std.npy'
    norm,mean,std=load_stats(stats_path)
    metadata=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),source_sha256=sha256(SOURCE),
                  torch_version=torch.__version__,device='cpu',normalization=norm,strict_state_tensors=tensors,
                  resize=[512,512],mask_rule='cv2 default bilinear resize then >0, matching existing code',
                  split='test',selection='all 148 supplied test samples; shortlist first for early inspection')
    metadata_path=OUT/'inference_metadata.json'
    if metadata_path.exists():
        old=json.loads(metadata_path.read_text())
        for key in ['checkpoint_sha256','source_sha256','normalization']:
            if old[key]!=metadata[key]: raise RuntimeError(f'Existing run has different {key}; use a separate output directory.')
    metadata_path.write_text(json.dumps(metadata,indent=2))
    (OUT/'predictions').mkdir(exist_ok=True)
    names=[n.strip() for n in (DATA/'test_list.txt').read_text().splitlines()]
    selected=list(csv.DictReader((ROOT/'reviewer9_audit'/'selected_test_examples.csv').open()))
    priority=[r['filename'] for r in selected]
    names=priority+[n for n in names if n not in priority]
    if args.limit: names=names[:args.limit]
    resultpath=OUT/'per_image_metrics.csv'
    rows=list(csv.DictReader(resultpath.open())) if resultpath.exists() else []
    done={r['filename'] for r in rows}
    with torch.inference_mode():
        for i,name in enumerate(names):
            if name in done: continue
            t=time.monotonic()
            rgb,ms,gt,nearest=prepare(name,mean,std)
            logits=model(rgb,ms)
            if not torch.isfinite(logits).all(): raise ValueError(f'Nonfinite output for {name}')
            pred=logits.argmax(1)[0].numpy().astype(bool)
            row=dict(filename=name,**scores(pred,gt),weed_pct=float(100*gt.mean()),predicted_weed_pct=float(100*pred.mean()),
                     nearest_gt_iou=scores(pred,nearest)['iou'],resize_gt_changed_pixels=int((gt!=nearest).sum()),
                     seconds=time.monotonic()-t)
            cv2.imwrite(str(OUT/'predictions'/(Path(name).stem+'.png')),pred.astype(np.uint8)*255)
            rows.append(row)
            with resultpath.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerows(rows)
            print(f"{len(rows)}/148 | {Path(name).stem.split('_lat_')[0]} | IoU {row['iou']:.4f} | recall {row['recall']:.4f} | {row['seconds']:.1f}s",flush=True)
    summary=dict(n=len(rows),provisional=norm['provisional'],
                 mean_per_image_iou=float(np.mean([float(r['iou']) for r in rows])),
                 mean_per_image_dice=float(np.mean([float(r['dice']) for r in rows])),
                 pooled_pixel_iou=sum(int(r['tp']) for r in rows)/sum(int(r['tp'])+int(r['fp'])+int(r['fn']) for r in rows),
                 note='One checkpoint. Not a reproduction of five-run paper averages; normalization provenance must be verified.')
    (OUT/'metrics_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
