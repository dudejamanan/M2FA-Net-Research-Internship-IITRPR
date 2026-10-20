# Six-case figure for the manuscript

Use the PDF for LaTeX to retain embedded, selectable text. The figure is 170 mm wide and approximately 234 mm high, intended for full-page width. Font sizes are 9.8-10.8 pt at that width. Avoid shrinking it to a single column. PNG and LZW-compressed TIFF are exported at 600 dpi.

## Suggested caption

Qualitative analysis of M²FA-Net on six challenging test images: (a) a sparse weed region, (b) small isolated weed regions, (c) spatially uneven brightness, (d) dense vegetation, (e) a darker canopy, and (f) a dense irregular canopy. Columns show the RGB image, binary ground truth, M²FA-Net prediction, and error overlay. Green, orange, and blue indicate true-positive, false-positive, and false-negative pixels, respectively. Foreground IoU and Dice are reported for each image. Panels (a) and (b) illustrate missed weed areas and false-positive detections, while panel (d) shows over-segmentation. The darker canopy in (e) is not a verified severe cast-shadow example.

## LaTeX insertion

```latex
\begin{figure*}[p]
    \centering
    \includegraphics[width=170mm]{Images/figure_qualitative_six_cases.pdf}
    \caption{Qualitative analysis of $M^2$FA-Net on six challenging test images: (a) a sparse weed region, (b) small isolated weed regions, (c) spatially uneven brightness, (d) dense vegetation, (e) a darker canopy, and (f) a dense irregular canopy. Columns show the RGB image, binary ground truth, $M^2$FA-Net prediction, and error overlay. Green, orange, and blue indicate true-positive, false-positive, and false-negative pixels, respectively. Foreground IoU and Dice are reported for each image. The darker canopy in (e) is not a verified severe cast-shadow example.}
    \label{fig:difficult_cases}
\end{figure*}
```

Place this six-row figure on a dedicated figure page. If the journal requires a substantially narrower figure width, split the layout into two groups of three rows instead of shrinking the font.

## Exact sample mapping

| Panel | Sample | Weed IoU | Dice |
|---|---|---|---|
| (a) | `DJI_DateTime_2024_06_04_09_25_0199_lat_10.3020021_lon_105.2721516_alt_20.218m.JPG` | 23.57% | 38.15% |
| (b) | `DJI_DateTime_2024_06_04_09_20_0119_lat_10.3011328_lon_105.2724032_alt_20.255m.JPG` | 48.57% | 65.38% |
| (c) | `DJI_DateTime_2024_06_04_11_07_0066_lat_10.3005700_lon_105.2715969_alt_20.223m.JPG` | 89.25% | 94.32% |
| (d) | `DJI_DateTime_2025_01_15_15_14_0019_lat_10.3632620_lon_105.4196284_alt_20.465m.JPG` | 71.78% | 83.58% |
| (e) | `DJI_DateTime_2025_01_15_14_56_0088_lat_10.3618167_lon_105.4227889_alt_12.000m.JPG` | 88.01% | 93.62% |
| (f) | `DJI_DateTime_2025_01_15_14_53_0022_lat_10.3622227_lon_105.4231242_alt_11.954m.JPG` | 88.09% | 93.67% |

The six cases are the condition-based shortlist selected before inference, not the six lowest-IoU samples. Predictions use the supplied checkpoint and original normalization. No baseline predictions were added. Display extent is restored to the native 4:3 aspect ratio for all columns; evaluation remains at 512 x 512 using the original code.
