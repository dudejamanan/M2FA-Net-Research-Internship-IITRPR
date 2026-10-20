from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reviewer9_audit'/'checkpoint_analysis'
path=ROOT/'ms_mean_std.npy'
supplied=np.load(path,allow_pickle=True).item()
reconstructed=json.loads((OUT/'reconstructed_normalization.json').read_text())
differences={k:float(np.max(np.abs(np.asarray(supplied[k])-np.asarray(reconstructed[k])))) for k in ['mean','std']}
if any(v!=0 for v in differences.values()):
    raise RuntimeError(f'Values differ; rerun required: {differences}')
meta=json.loads((OUT/'inference_metadata.json').read_text())
meta['normalization']={k:np.asarray(supplied[k]).tolist() for k in ['mean','std']}
meta['normalization'].update(source=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),provisional=False)
meta['normalization_verification']={'original_file_added_during_inference':True,'exact_match_to_reconstruction':True,'max_abs_difference':differences,'rerun_needed':False}
# This script changes only provenance; it does not change masks or their measured scores.
(OUT/'inference_metadata.json').write_text(json.dumps(meta,indent=2))
summary=OUT/'metrics_summary.json'
if summary.exists():
    s=json.loads(summary.read_text());s['provisional']=False
    s['note']='One supplied checkpoint with original normalization verified. Not a reproduction of five-run paper averages.'
    summary.write_text(json.dumps(s,indent=2))
print('Original normalization verified: exact numerical match. Metadata updated.')
