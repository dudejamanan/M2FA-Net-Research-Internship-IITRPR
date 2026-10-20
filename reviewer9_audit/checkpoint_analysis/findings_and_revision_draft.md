# Reviewer 9: measured checkpoint analysis

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
| Sparse, edge-truncated weed region | `2024_06_04_09_25_0199` | 23.57 |
| Small isolated weed regions | `2024_06_04_09_20_0119` | 48.57 |
| Bright patch / uneven appearance | `2024_06_04_11_07_0066` | 89.25 |
| Dense mixed canopy candidate | `2025_01_15_15_14_0019` | 71.78 |
| Dark canopy / self-shadow candidate | `2025_01_15_14_56_0088` | 88.01 |
| Dense irregular canopy | `2025_01_15_14_53_0022` | 88.09 |

The two sparse cases are especially useful because they combine the requested small-region condition with real segmentation errors. In `2024_06_04_09_25_0199`, the predicted weed area covers only part of the edge-truncated annotated region (recall 23.6%). In `2024_06_04_09_20_0119`, an annotated isolated region is missed and a false positive appears near the upper-left image edge. `2025_01_15_15_14_0019` shows substantial over-segmentation in a dense canopy (precision 75.1%, recall 94.2%). These are observed errors against the benchmark masks; their physical cause cannot be proved from this comparison.

The bright-patch example and darker canopy example are useful alongside these failures because the model performs better on those particular images, while still making local errors. Do not infer condition-wide robustness from one selected image. The darker canopy example does not establish robustness to strong cast shadows. Binary weed/background masks do not independently label crop leaves or occlusion.

## Ten lowest-IoU cases

Scores below are percentages; the subset was deliberately selected for poor performance and does not estimate overall performance.

| Sample | Weed IoU | Dice | Precision | Recall |
|---|---|---|---|---|
| `2024_09_30_10_34_0042` | 18.97 | 31.89 | 96.2 | 19.1 |
| `2024_06_04_09_25_0199` | 23.57 | 38.15 | 100.0 | 23.6 |
| `2024_06_04_09_20_0120` | 38.37 | 55.46 | 85.2 | 41.1 |
| `2024_09_30_10_33_0032` | 44.03 | 61.14 | 62.0 | 60.3 |
| `2024_06_04_09_32_0313` | 48.29 | 65.13 | 60.3 | 70.9 |
| `2024_06_04_09_20_0119` | 48.57 | 65.38 | 62.2 | 69.0 |
| `2024_06_04_09_32_0319` | 50.59 | 67.19 | 88.8 | 54.1 |
| `2024_09_30_10_33_0031` | 50.76 | 67.34 | 85.6 | 55.5 |
| `2024_06_02_13_43_0046` | 51.98 | 68.40 | 71.7 | 65.4 |
| `2024_06_02_13_47_0116` | 52.44 | 68.80 | 56.4 | 88.1 |

## Aggregate scores and their interpretation

| Aggregation over this one checkpoint | Foreground IoU (%) | Foreground Dice (%) |
|---|---|---|
| Mean of individual-image scores | 77.16 | 86.34 |
| All test pixels pooled | 84.28 | 91.47 |
| Mean of batch scores, batch size 4, original test order | 79.75 | 88.50 |

The local `evaluate()` function computes a score over all pixels in each batch, then averages those batch scores. Its test loader uses batch size 4. That quantity is distinct from a mean of image scores or a globally pooled score. The supplied weights are one checkpoint with unspecified run identity; these results must not replace or be called a reproduction of the paper's five-run averages (80.79% IoU, 89.21% Dice). The table makes the aggregation explicit so that a change in metric definition is not mistaken for a change in model performance.

There are 19 test images with native annotated weed coverage below 5%; their mean per-image IoU is 57.34%. This is a descriptive subset, confounded by other scene properties. A small-region-specific causal explanation would need more evidence.

## Preprocessing and verification

The audit uses the supplied model's evaluation path: RGB ImageNet normalization; G/R/RE/NIR channel order; spectral scaling by 65535; NDVI/NDRE mapped to [0,1]; original channel means and standard deviations; 512 x 512 inputs; evaluation mode; and argmax outputs without post-processing. Backbones are created without a pretrained download, then every parameter and buffer is replaced through strict checkpoint loading. Model classes are extracted from the local training source without executing its training entry point. CPU float32 inference is used.

The existing code resizes binary GT with OpenCV's default bilinear interpolation and thresholds it at >0. That can expand boundaries compared with nearest-neighbor mask resizing. This audit retains that rule for consistency, and also records a nearest-neighbor-GT IoU sensitivity column. The nearest-neighbor-GT mean image IoU is 76.86%. Do not silently change the original evaluation rule only for the new figures.

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
