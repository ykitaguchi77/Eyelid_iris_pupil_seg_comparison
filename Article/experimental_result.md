
## Methods

### Study objective
We investigate whether incorporating *occluded* regions (occ) in iris/pupil segmentation enables **accurate and efficient inference** with simpler post-processing. Prior work includes (i) directly regressing ellipse parameters for iris/pupil (Method1) and (ii) segmenting edges followed by ellipse fitting (Method2). In this study, we focus on a segmentation-based pipeline and evaluate whether **full-mask–based ellipse fitting (FullMax)** improves performance over **exposed-only fitting (OuterArc)**.

### Data and cross-validation protocol
We used **5-fold cross-validation** with an image resolution of **512×512**. The full dataset contains **1992 images** from **122 patients/subjects**. In each fold, models were trained on **4/5 folds** and evaluated on the remaining **1/5 held-out fold** (out-of-fold evaluation), such that each image is tested exactly once.

**Train/test patient counts.** Per fold, the number of training patients was **97–98**, and the number of test patients was **24–25**, with **no patient overlap** between train and test within each fold.

**Subject clustering.** The subject identifier is defined as the substring **before the first hyphen** in the image filename (e.g., `162-...` → subject `162`). Multiple images from the same subject are **not independent**; therefore, all statistical comparisons are performed at the **subject level** by averaging per-image Dice scores within each subject. Across the full cross-validation evaluation, the number of unique test subjects was **n = 122**.

### Models evaluated
To match the presentation order in the Results section, we describe the U-Net methods first, followed by the ablation studies (YOLO11l-seg, SegFormer-B2).

#### U-Net (Methods 1–3)
- **Method1 (ellipse regression)**: a U-Net model that predicts eyelid segmentation and directly regresses ellipse parameters for iris and pupil.
- **Method2 (edge → fitting)**: a U-Net model that predicts eyelid/iris/pupil edges, followed by ellipse fitting.
- **Method3 (6-class region segmentation)**: a U-Net model that predicts 6 region classes: background, conjunctiva/eyelid region, iris-visible, iris-occluded, pupil-visible, pupil-occluded.

#### Ablation model: YOLO11l-seg (segmentation)
We evaluated a YOLO11l-seg–based instance segmentation model for multi-class eye component segmentation and converted its predicted masks into class-wise binary masks for eyelid, iris, and pupil.

#### Ablation model: SegFormer-B2 (semantic segmentation)
We evaluated a SegFormer-B2 semantic segmentation model (initialized from `nvidia/segformer-b2-finetuned-ade-512-512` with a 6-class decoder head) under the same 5-fold protocol and post-processing definitions as Method3/YOLO, enabling direct comparison across architectures.

### Post-processing methods (compared)
We compare post-processing strategies that convert predicted masks into ellipse masks for iris and pupil. These post-processing variants are applied to **Method3 (U-Net 6-class outputs)**, **YOLO11l-seg outputs**, and **SegFormer outputs** (i.e., when visible/occluded masks are available).

- **Raw (no ellipse fitting)**: use union masks directly  
  - iris = (iris_visible ∪ iris_occluded)  
  - pupil = (pupil_visible ∪ pupil_occluded)

- **OuterArc (exposed-arc ellipse fit; no RANSAC)**:  
  1) form full mask = (visible ∪ occluded)  
  2) extract the **maximum external contour** of the full mask  
  3) remove points near the visible/occluded boundary (often aligned with eyelid cut)  
  4) keep outer contour points close to the **visible region** and fit an ellipse with `cv2.fitEllipse`

- **FullMax (full-mask max-contour ellipse fit; no RANSAC)**:  
  1) form full mask = (visible ∪ occluded)  
  2) extract the **maximum external contour**  
  3) fit an ellipse to all contour points with `cv2.fitEllipse`

- **RANSAC (whole-mask)** *(U-Net Method3 / YOLO11l / SegFormer)*:  
  extract the maximum external contour of the full mask and fit an ellipse using RANSAC (skimage `ransac` + `EllipseModel`).

In all modes, the eyelid mask is unchanged (ellipse fitting is applied only to iris and pupil), hence eyelid metrics are identical across post-processing variants within the same model.

### Evaluation metric
We report the Dice similarity coefficient for:
- **Eyelid**
- **Iris**
- **Pupil**
- **Mean Dice** (average of the three class Dice scores)

### Statistical analysis (recommended)
All hypothesis tests are performed on **subject-level mean Dice** (clustered by subject).

- **Effect size**: mean paired difference \(\Delta = A - B\) in subject-level mean Dice
- **Uncertainty**: subject-level paired bootstrap **95% confidence interval (CI)**
- **Significance**: subject-level paired **permutation test** (sign-flip), two-sided
- **Multiple comparisons**: Holm correction (family-wise error rate control)

Permutation p-values may appear as 0.0 when no permuted statistic exceeds the observed statistic; these should be interpreted as \(p < 1/N_{\mathrm{perm}}\) (here, approximately \(p < 5 \times 10^{-5}\) for 20,000 permutations).

## Results

### U-Net (Methods 1–3)

#### Fold-average performance: Method1 vs Method2 vs Method3
We first compare the three established U-Net pipelines under 5-fold cross-validation (mean±std across folds):

| Method | Eyelid Dice (mean±std) | Iris Dice (mean±std) | Pupil Dice (mean±std) | Mean Dice (mean±std) |
|---|---|---|---|---|
| Method1 (ellipse regression) | 0.9845±0.0022 | 0.8879±0.0031 | 0.7225±0.0373 | 0.8649±0.0121 |
| Method2 (edge → fitting) | 0.9513±0.0132 | 0.8958±0.0107 | 0.8859±0.0230 | 0.9110±0.0138 |
| Method3 (6-class segmentation) | 0.9804±0.0022 | 0.9343±0.0086 | 0.9020±0.0133 | **0.9389±0.0064** |

**Interpretation.** Among U-Net approaches, Method3 achieved the highest mean Dice, motivating further analysis of post-processing within Method3.

### U-Net Method3: post-processing ablation (full vs exposed)
To test whether using occluded predictions (full mask = visible ∪ occluded) improves ellipse extraction stability, we evaluated post-processing variants on Method3 outputs and performed subject-clustered inference.

#### Fold-average performance (reference)
Mean Dice averaged over folds (reference):

| Post-processing | Mean Dice |
|---|---:|
| Raw | 0.921180 |
| OuterArc (exposed-only fit) | 0.935549 |
| FullMax (full-mask max-contour fit) | **0.943696** |
| RANSAC (whole-mask) | 0.938908 |

#### Subject-level ranking (recommended)

| Post-processing | Subject mean Dice | n_subjects |
|---|---:|---:|
| FullMax | **0.939232** | 122 |
| RANSAC (whole) | 0.934093 | 122 |
| OuterArc | 0.930789 | 122 |
| Raw | 0.917460 | 122 |

#### Direct comparison: full vs exposed (primary evidence within Method3)

| Comparison (A − B) | n_subjects | mean_diff | 95% CI | win_rate | p_perm | p_holm |
|---|---:|---:|---|---:|---:|---:|
| FullMax − OuterArc | 122 | +0.008443 | [+0.005363, +0.012299] | 0.836 | <5e-5 | <5e-5 |

**Interpretation.** On U-Net Method3 outputs, FullMax significantly outperformed exposed-only fitting (OuterArc), supporting the hypothesis that leveraging occluded regions improves geometric stability and downstream segmentation accuracy.

### Ablation study: YOLO11l-seg

#### Fold-average performance (reference)
Mean Dice averaged over folds (reference only; not used for hypothesis testing):

| Method | Mean Dice |
|---|---:|
| Raw | 0.932663 |
| OuterArc | 0.952667 |
| FullMax | **0.957827** |
| RANSAC (whole) | 0.956790 |
| RANSAC (arc) | 0.949780 |

#### Subject-level ranking (recommended)
Subject-level mean Dice (n=122 subjects):

| Method | Subject mean Dice | n_subjects |
|---|---:|---:|
| FullMax | **0.950497** | 122 |
| RANSAC (whole) | 0.949442 | 122 |
| OuterArc | 0.945110 | 122 |
| RANSAC (arc) | 0.942759 | 122 |
| Raw | 0.926026 | 122 |

#### Direct comparison: full vs exposed (primary evidence)
We directly compared full-mask–based inference/post-processing against exposed-only fitting at the subject level:

| Comparison (A − B) | n_subjects | mean_diff | 95% CI | win_rate | p_perm | p_holm |
|---|---:|---:|---|---:|---:|---:|
| FullMax − OuterArc | 122 | +0.005387 | [+0.002681, +0.008507] | 0.844 | <5e-5 | <5e-5 |
| RANSAC(whole) − RANSAC(arc) | 122 | +0.006683 | [+0.003985, +0.009717] | 0.820 | <5e-5 | <5e-5 |

**Interpretation.** Incorporating occluded regions (full mask) yields a statistically significant improvement over exposed-only fitting, and substantially simplifies post-processing by avoiding noisy eyelid-cut boundaries.

### Ablation study: SegFormer-B2

#### Fold-average performance (reference)
Mean Dice averaged over folds (reference only):

| Method | Mean Dice |
|---|---:|
| Raw | 0.938383 |
| OuterArc | 0.951834 |
| FullMax | **0.960701** |
| RANSAC (whole) | 0.958817 |
| RANSAC (arc) | 0.947079 |

#### Subject-level ranking (recommended)
Subject-level mean Dice (n=122 subjects):

| Method | Subject mean Dice | n_subjects |
|---|---:|---:|
| FullMax | **0.955841** | 122 |
| RANSAC (whole) | 0.953984 | 122 |
| OuterArc | 0.944337 | 122 |
| RANSAC (arc) | 0.939781 | 122 |
| Raw | 0.934034 | 122 |

#### Direct comparison: full vs exposed (primary evidence)

| Comparison (A − B) | n_subjects | mean_diff | 95% CI | win_rate | p_perm | p_holm |
|---|---:|---:|---|---:|---:|---:|
| FullMax − OuterArc | 122 | +0.011504 | [+0.006847, +0.017177] | 0.787 | <5e-5 | <5e-5 |
| RANSAC(whole) − RANSAC(arc) | 122 | +0.014203 | [+0.008433, +0.021340] | 0.705 | <5e-5 | <5e-5 |

**Interpretation.** The SegFormer ablation reproduced the same qualitative finding: full-mask–based ellipse extraction (FullMax) consistently outperformed exposed-only fitting (OuterArc) under subject-clustered inference, supporting the model-independent benefit of incorporating occluded region predictions.

## Summary of findings
Across U-Net Methods 1–3, Method3 (6-class segmentation) achieved the best mean Dice. Within Method3, **FullMax** significantly outperformed exposed-only fitting (**OuterArc**). As ablation studies on **YOLO11l-seg** and **SegFormer-B2**, we observed the same qualitative trend: using full (visible + occluded) predictions improves ellipse extraction and simplifies post-processing while yielding statistically significant gains under subject-clustered inference.

