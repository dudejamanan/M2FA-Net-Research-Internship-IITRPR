# Reviewer 9: local dataset audit and revision plan

**Checkpoint follow-up:** The user subsequently supplied `best_model_MultiLevelFusionDeepLab.pth` and the original `ms_mean_std.npy`. The complete measured test analysis and updated revision text are in [checkpoint_analysis/findings_and_revision_draft.md](checkpoint_analysis/findings_and_revision_draft.md). The absence-of-checkpoint observations below describe the initial dataset-only audit.

This audit inspected all 734 RGB images and corresponding binary masks numerically, and visually reviewed contact sheets covering all 148 test images. Three candidates were also inspected at original resolution. Split counts: 438 train, 148 validation, 148 test. The manuscript and original files were not modified.

## What the available evidence supports

- Small/sparse annotated weed regions: confirmed. Nineteen test images have positive weed coverage below 5%; one is below 1%. Low total foreground area and small individual regions are different properties. The 0.44% example contains one region touching the left image edge, so pair it with the 2.02% example containing four isolated regions. Do not call these subpixel weeds or claim leaf-level annotations.
- Dense vegetation and irregular weed/background interfaces: confirmed. January images offer visually dense canopies. Describe them as dense mixed vegetation or complex interfaces. Exact crop-over-weed occlusion cannot be established from a binary weed/background mask, which contains neither a separate crop class nor occlusion labels. High weed coverage alone does not quantify overlap.
- Uneven brightness / bright patches: confirmed visually, especially in June 4 images captured around 11:00. Possible glare or hotspot appearance is visible; its optical cause is unverified. Describe the visible appearance rather than claiming measured illumination conditions. Differences across images are confounded by vegetation, altitude, exposure, and acquisition date.
- Strong shadows: not confidently established. The January 14:56 examples have dark canopy areas consistent with self-shadowing. Dark-pixel ranking is a screening proxy, not a shadow detector. The darkest test example has about 24% of RGB grayscale pixels below 50/255, but this is not a shadow-area estimate. Do not label it a verified strong cast-shadow case. This audit cannot establish absence of strong shadows from the entire training/validation set, which was not exhaustively visually reviewed.
- Actual model failures: visible partial failures already occur in the saved full-model Run 1 figure. Rows marked RGB Image (1), (10), and (20) show holes/missing interior areas, broken or missing narrow connections, and under-segmentation respectively. These observations are against the displayed annotations; they are not newly measured per-image scores.

## Shortlisted test images

The preview displays original RGB appearance next to the binary ground truth. All rows below are in the supplied test split. Filenames are preserved in selected_test_examples.csv.

| Candidate | Filename | Annotated weed coverage |
|---|---|---|
| Sparse, edge-truncated weed region | `DJI_DateTime_2024_06_04_09_25_0199_lat_10.3020021_lon_105.2721516_alt_20.218m.JPG` | 0.44% |
| Small isolated weed regions | `DJI_DateTime_2024_06_04_09_20_0119_lat_10.3011328_lon_105.2724032_alt_20.255m.JPG` | 2.02% |
| Bright patch / uneven appearance | `DJI_DateTime_2024_06_04_11_07_0066_lat_10.3005700_lon_105.2715969_alt_20.223m.JPG` | 56.89% |
| Dense mixed canopy candidate | `DJI_DateTime_2025_01_15_15_14_0019_lat_10.3632620_lon_105.4196284_alt_20.465m.JPG` | 40.62% |
| Dark canopy / self-shadow candidate | `DJI_DateTime_2025_01_15_14_56_0088_lat_10.3618167_lon_105.4227889_alt_12.000m.JPG` | 82.09% |
| Dense irregular canopy | `DJI_DateTime_2025_01_15_14_53_0022_lat_10.3622227_lon_105.4231242_alt_11.954m.JPG` | 84.17% |

## Important provenance of existing predictions

The local script `Multi Level Fusion/MultiLevelFusion.py.py` calls `save_ablation_predictions(..., dataset=val_dataset, ...)` at lines 969-975. Its default indices are [1, 5, 10, 15, 20], zero-based. Thus the saved ablation grids should be treated as validation visualizations, subject to confirmation that this exact script produced them. Their titles are indices, not unique filenames. Do not present them as test cases.

The inspected grid is `Results/Ablation study results/ablation_study_results_20epochs_20260615_150815-20260623T121140Z-3-001/ablation_study_results_20epochs_20260615_150815/FULL_MODEL_(Proposed)_Run_1/sample_predictions.png`.

No .pth, .pt, .ckpt, or .onnx files were found in the workspace search. No complete set of individual test prediction masks was found. New test predictions and a true worst-case ranking therefore remain to be produced from the trained weights, ideally on the original training machine. Saved composite figures are insufficient for reliable full-test scoring.

## Recommended manuscript changes

1. Add a dedicated difficult-case comparison, approximately six rows, with RGB, ground truth, two strong baselines (e.g. Swin and SegFormer, as supported by the paper's results), M2FA-Net, and a clearly explained TP/FP/FN overlay. Use identical images, preprocessing, and checkpoints selected by validation performance for all methods. Preserve original RGB display and use consistent mask colors.
2. Include sparse regions, dense vegetation, brightness nonuniformity, and darker canopy examples from the shortlist. Add matched zoomed insets where details are lost at page size; identify their locations on the full image.
3. Add two or three explicit failure examples selected from frozen test predictions: low per-image foreground IoU, missed small components, false positive regions, or merged/broken boundaries. State the selection rule and selection bias. These are an illustrative subset, not a representative estimate of performance. Keep all aggregate results on the complete unchanged test split.
4. Prefer some cases where the proposed model is comparable to, or worse than, a baseline if such cases occur. Do not claim this happens until measured. Discuss observed failure types and qualify causal explanations as hypotheses.
5. Report sample IDs and per-image weed IoU/Dice alongside failures. Optional supplementary table: number of manually tagged cases per condition and performance across ALL tagged cases. Conditions may overlap. Avoid claims of strong-shadow robustness when there are too few verified examples.
6. Replace the current broad claim that M2FA-Net accurately detects both large and small weed regions with an account that also acknowledges missed regions and boundary errors visible in the new figure.

No retraining is inherently needed to answer this comment. Inference with existing trained weights and a more candid qualitative analysis may be sufficient. Do not synthesize shadows and present them as natural dataset examples. If adding synthetic brightness/shadow perturbations, label them as a separate controlled stress test; an RGB-only perturbation tests RGB-branch sensitivity, not a physically complete multispectral illumination change.

## Draft reply: use only AFTER generating and inspecting the new figure

Thank you for this suggestion. We have expanded the qualitative evaluation in Fig. [X] to include test samples with sparse and small weed regions, dense vegetation and complex weed/background interfaces, and spatially nonuniform brightness. We additionally include [N] failure cases showing [insert errors actually observed in the selected predictions]. The revised figure provides RGB inputs, ground-truth annotations, baseline and M2FA-Net predictions, and error visualizations with enlarged regions of interest. The accompanying discussion now describes both successful segmentation and remaining errors. We also clarify that the available benchmark does not provide verified strong-shadow annotations; consequently, the darker canopy examples are not used to claim general robustness to strong cast shadows. All examples use the original test partition, and the complete-test quantitative evaluation is unchanged.

If no unambiguous natural shadow example can be verified, add: "A dedicated evaluation under severe cast shadows remains a limitation and will require additional annotated acquisitions."

## Draft discussion structure

The additional qualitative examples cover sparse weed regions, dense vegetation, and spatially nonuniform brightness. [Describe the comparisons observed for each selected row.] Despite its overall performance, M2FA-Net exhibits [verified false negatives/false positives/boundary errors] in [sample IDs]. [Describe locations, extent, and baseline behavior.] These examples indicate that the method remains imperfect under the depicted conditions. The binary annotations do not establish leaf-level occlusion, and the available examples do not support a separate claim of robustness to severe cast shadows.

## Separate metadata inconsistency noticed

The manuscript says that per-sample acquisition dates are unspecified, but the local metadata CSV and standardized filenames contain dates, times, coordinates, and altitude. This does not prove independence of fields or flights, but the statement that acquisition date is unavailable should be corrected. Date alone is also not an illumination-condition label.
