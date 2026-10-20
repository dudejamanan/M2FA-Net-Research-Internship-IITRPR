from pathlib import Path
import csv,json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reviewer9_audit'/'checkpoint_analysis'
DATA=ROOT/'Dataset'/'WeedyRice-RGBMS-DB'
rows=list(csv.DictReader((OUT/'per_image_metrics.csv').open()))
lookup={r['filename']:r for r in rows}
names=[n.strip() for n in (DATA/'test_list.txt').read_text().splitlines()]
assert len(rows)==len(set(lookup))==148
assert set(lookup)==set(names),'Prediction set differs from supplied test set'
metadata=json.loads((OUT/'inference_metadata.json').read_text())
assert metadata['normalization']['provisional'] is False,'Verify original normalizer first'
for name in names:
    r=lookup[name]
    pred=cv2.imread(str(OUT/'predictions'/(Path(name).stem+'.png')),0)
    assert pred is not None and pred.shape==(512,512)
    assert set(np.unique(pred)).issubset({0,255})
    gt=cv2.resize(cv2.imread(str(DATA/'Masks'/(Path(name).stem+'.png')),0),(512,512))>0
    pred=pred>0
    tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum())
    assert (tp,fp,fn)==tuple(int(r[k]) for k in ['tp','fp','fn'])
    assert abs(tp/(tp+fp+fn)-float(r['iou']))<1e-12
    assert abs(2*tp/(2*tp+fp+fn)-float(r['dice']))<1e-12

def mean(metric,items=rows): return float(np.mean([float(r[metric]) for r in items]))
def pooled(metric,items):
    tp=sum(int(r['tp']) for r in items);fp=sum(int(r['fp']) for r in items);fn=sum(int(r['fn']) for r in items)
    return tp/(tp+fp+fn) if metric=='iou' else 2*tp/(2*tp+fp+fn)
def batchmean(metric,batchsize):
    return float(np.mean([pooled(metric,[lookup[n] for n in names[i:i+batchsize]]) for i in range(0,len(names),batchsize)]))
audit={r['filename']:r for r in csv.DictReader((ROOT/'reviewer9_audit'/'image_audit.csv').open())}
sparse=[r for r in rows if 0<float(audit[r['filename']]['weed_pct'])<5]
summary=dict(n=148,provisional=False,mean_per_image_iou=mean('iou'),mean_per_image_dice=mean('dice'),
             pooled_pixel_iou=pooled('iou',rows),pooled_pixel_dice=pooled('dice',rows),
             mean_batch4_iou=batchmean('iou',4),mean_batch4_dice=batchmean('dice',4),
             nearest_gt_mean_per_image_iou=mean('nearest_gt_iou'),
             native_weed_below5pct_count=len(sparse),native_weed_below5pct_mean_iou=mean('iou',sparse),
             note='One supplied checkpoint with original normalization verified. Different aggregation rules are distinct metrics; none is a five-run paper result.',
             verification='All 148 saved binary masks independently rescored against original-code resized test GT; counts and scores agree.')
(OUT/'metrics_summary.json').write_text(json.dumps(summary,indent=2))
worst=sorted(rows,key=lambda r:float(r['iou']))
def sid(r): return Path(r['filename']).stem.replace('DJI_DateTime_','').split('_lat_')[0]
table='\n'.join(f"| `{sid(r)}` | {100*float(r['iou']):.2f} | {100*float(r['dice']):.2f} | {100*float(r['precision']):.1f} | {100*float(r['recall']):.1f} |" for r in worst[:10])
selected=list(csv.DictReader((ROOT/'reviewer9_audit'/'selected_test_examples.csv').open()))
conditiontable='\n'.join(f"| {s['condition']} | `{sid(lookup[s['filename']])}` | {100*float(lookup[s['filename']]['iou']):.2f} |" for s in selected)
report=f'''# Reviewer 9: measured checkpoint analysis

Completed inference on all **148 supplied test images**, using `best_model_MultiLevelFusionDeepLab.pth` and the original supplied `ms_mean_std.npy`. All 703 tensors load strictly into the local model implementation. The original means and standard deviations exactly match the values reconstructed at the start of this analysis (maximum absolute difference zero); hence the predictions made before that file arrived do not require rerunning. These results are no longer provisional with respect to normalization.

## Deliverables

- `difficult_test_cases.png`: six condition-based cases selected BEFORE inference, with RGB, GT, prediction, and errors.
- `lowest_iou_cases.png`: six lowest-foreground-IoU test cases from the complete 148-image ranking.
- `failure_zoom_details.png`: full-image crop locations and enlarged missed regions for the three lowest-IoU cases.
- `predictions/`: all 148 binary masks at 512 x 512.
- `ranked_test_cases.csv`: all test cases ranked by foreground IoU; complete source filenames preserved.
- `inference_metadata.json`: checkpoint and source hashes, normalization values and provenance, and preprocessing.

## Concrete evidence for the response

| Condition | Short sample ID | Weed IoU (%) |
|---|---|---|
{conditiontable}

The two sparse cases are especially useful because they combine the requested small-region condition with real segmentation errors. In `2024_06_04_09_25_0199`, the predicted weed area covers only part of the edge-truncated annotated region (recall 23.6%). In `2024_06_04_09_20_0119`, an annotated isolated region is missed and a false positive appears near the upper-left image edge. `2025_01_15_15_14_0019` shows substantial over-segmentation in a dense canopy (precision 75.1%, recall 94.2%). These are observed errors against the benchmark masks; their physical cause cannot be proved from this comparison.

The bright-patch example and darker canopy example are useful alongside these failures because the model performs better on those particular images, while still making local errors. Do not infer condition-wide robustness from one selected image. The darker canopy example does not establish robustness to strong cast shadows. Binary weed/background masks do not independently label crop leaves or occlusion.

## Ten lowest-IoU cases

Scores below are percentages; the subset was deliberately selected for poor performance and does not estimate overall performance.

| Sample | Weed IoU | Dice | Precision | Recall |
|---|---|---|---|---|
{table}

## Aggregate scores and their interpretation

| Aggregation over this one checkpoint | Foreground IoU (%) | Foreground Dice (%) |
|---|---|---|
| Mean of individual-image scores | {100*summary['mean_per_image_iou']:.2f} | {100*summary['mean_per_image_dice']:.2f} |
| All test pixels pooled | {100*summary['pooled_pixel_iou']:.2f} | {100*summary['pooled_pixel_dice']:.2f} |
| Mean of batch scores, batch size 4, original test order | {100*summary['mean_batch4_iou']:.2f} | {100*summary['mean_batch4_dice']:.2f} |

The local `evaluate()` function computes a score over all pixels in each batch, then averages those batch scores. Its test loader uses batch size 4. That quantity is distinct from a mean of image scores or a globally pooled score. The supplied weights are one checkpoint with unspecified run identity; these results must not replace or be called a reproduction of the paper's five-run averages (80.79% IoU, 89.21% Dice). The table makes the aggregation explicit so that a change in metric definition is not mistaken for a change in model performance.

There are {len(sparse)} test images with native annotated weed coverage below 5%; their mean per-image IoU is {100*summary['native_weed_below5pct_mean_iou']:.2f}%. This is a descriptive subset, confounded by other scene properties. A small-region-specific causal explanation would need more evidence.

## Preprocessing and verification

The audit uses the supplied model's evaluation path: RGB ImageNet normalization; G/R/RE/NIR channel order; spectral scaling by 65535; NDVI/NDRE mapped to [0,1]; original channel means and standard deviations; 512 x 512 inputs; evaluation mode; and argmax outputs without post-processing. Backbones are created without a pretrained download, then every parameter and buffer is replaced through strict checkpoint loading. Model classes are extracted from the local training source without executing its training entry point. CPU float32 inference is used.

The existing code resizes binary GT with OpenCV's default bilinear interpolation and thresholds it at >0. That can expand boundaries compared with nearest-neighbor mask resizing. This audit retains that rule for consistency, and also records a nearest-neighbor-GT IoU sensitivity column. The nearest-neighbor-GT mean image IoU is {100*summary['nearest_gt_mean_per_image_iou']:.2f}%. Do not silently change the original evaluation rule only for the new figures.

Every saved mask was independently rescored from disk: all 148 outputs are binary, have the expected dimensions, match the supplied test IDs exactly, and reproduce the stored TP/FP/FN, IoU, and Dice. Checkpoint loading verifies tensor compatibility, but the exact training run and best-epoch selection cannot be recovered from a bare state dictionary.

## How to revise the paper

1. Add the condition-based panel and a smaller dedicated failure panel. Include matched zooms for the small regions. Use the same case IDs for any additional baseline predictions.
2. To extend the **comparison**, generate Swin/SegFormer or other selected baseline predictions on these same images using their original trained checkpoints and preprocessing. Only the proposed-model checkpoint was supplied here, so these figures currently provide the proposed model's qualitative analysis, not a newly completed baseline comparison.
3. Explain missed small regions, false positives, and over-segmentation with case IDs. Do not claim that all failures are caused by illumination or overlap; the figure demonstrates errors, not causation.
4. Keep strong cast-shadow robustness as an unverified condition and acknowledge the need for additional annotated acquisitions if no clear natural example is available.

## Suggested caption for the new proposed-model panel

Qualitative analysis of M2FA-Net on six test images selected to illustrate sparse weed regions, dense vegetation with complex interfaces, spatially nonuniform brightness, and a darker canopy. Columns show RGB inputs, binary ground truth, model predictions, and error overlays. Green, orange, and blue indicate true-positive, false-positive, and false-negative pixels, respectively. Per-image foreground IoU and Dice are reported above each row. The sparse-region examples show missed annotated regions and false positives, while the dense-canopy example illustrates over-segmentation. The darker canopy image is not a verified severe cast-shadow case.

## Suggested manuscript paragraph

The expanded qualitative analysis includes difficult test samples and explicit failure cases. In the sparse-region sample 2024_06_04_09_25_0199, M2FA-Net detects only part of the annotated weed region, yielding a foreground IoU of 23.57%. In sample 2024_06_04_09_20_0119, missed portions of isolated weed regions and a false-positive area reduce IoU to 48.57%. In the dense-canopy sample 2025_01_15_15_14_0019, the model exhibits over-segmentation, with precision of 75.1% despite recall of 94.2%. These cases show that the method remains imperfect on sparse targets and complex weed/background interfaces. The examples with bright patches and darker canopy areas illustrate performance under the depicted appearances, but do not establish general robustness to severe cast shadows or independently verified illumination changes.

## Reviewer reply draft: finalize after integrating figures and any baseline columns

Thank you for this suggestion. We have expanded the qualitative evaluation to cover test images with sparse and small weed regions, dense vegetation and complex weed/background interfaces, spatially nonuniform brightness, and darker canopy areas. The added figures include explicit failure examples, error overlays, enlarged regions of interest, and per-image foreground IoU/Dice. We now discuss missed weed regions, false-positive detections, and over-segmentation rather than presenting only favorable examples. Failure examples were selected from the lowest-IoU cases among all 148 test images, while the condition-based examples were selected before inference. We also clarify that the available examples do not establish performance under verified severe cast shadows, which remains a limitation. [After adding baseline columns, describe the exact new comparisons here.] The figures use the existing trained checkpoint without retraining or changing the supplied test split.
'''
(OUT/'findings_and_revision_draft.md').write_text(report,encoding='utf-8')
print(json.dumps(summary,indent=2))
print('All saved masks verified; final findings and revision draft saved.')

if __name__=='__main__': pass
