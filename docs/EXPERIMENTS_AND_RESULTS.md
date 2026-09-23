# Experiments, Evaluation, and Empirical Results

## 1. Dataset Characteristics & Partitioning

### 1.1 Source & Composition
- **Benchmark Source**: Kaggle public dataset [`pythonafroz/solar-panel-images`](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images).
- **Total Valid Images**: 885 images across six canonical conditions.
- **Corrupted / Unreadable Files**: 0.
- **Image File Formats**: 877 `.jpg`, 5 `.png`, 3 `.jpeg`.
- **Color Spaces**: 866 RGB (3-channel), 19 RGBA (4-channel; alpha-composited over white background during loading).
- **Spatial Resolution Range**:
  - Width: 149 px to 6,240 px (mean: 1,003.7 px, median: 720.0 px).
  - Height: 110 px to 5,376 px (mean: 809.4 px, median: 631.0 px).
  - Aspect Ratio: 0.51 to 3.60 (mean: 1.32, median: 1.33 $\approx$ 4:3).

### 1.2 Class Distribution
The dataset exhibits natural real-world class imbalance, with surface soiling and normal panels appearing far more frequently than physical fractures or internal electrical burns:

| Class Name | Total Sample Count | Proportion of Dataset | Visual Characteristics |
| :--- | :---: | :---: | :--- |
| **Bird-drop** | 207 | 23.39% | Localized white/gray organic matter patches |
| **Clean** | 193 | 21.81% | Unobstructed, defect-free photovoltaic glass |
| **Dusty** | 190 | 21.47% | Diffuse sand and particulate accumulation |
| **Snow-Covered** | 123 | 13.90% | Extensive white opaque snow cover |
| **Electrical-damage** | 103 | 11.64% | Localized cell burning, hot-spot browning, solder melting |
| **Physical-damage** | 69 | 7.80% | Glass spiderweb cracking, cell fracture lines, delamination |
| **Total** | **885** | **100.0%** | |

### 1.3 Leakage Prevention via Hash-Deduplicated Partitioning
Preliminary exploratory data analysis revealed duplicate and near-duplicate images within the raw dataset. Randomly splitting such images across training and test sets would cause data leakage, artificially inflating test evaluation metrics.

To eliminate leakage:
1. Exact file content hashes (SHA-256 and MD5) were calculated for all 885 images.
2. Identical duplicate hashes were grouped together into single atomic sample units.
3. A **hash-grouped stratified split** was executed with fixed random seed (seed = 42), allocating approximately 60% of data to training, 20% to validation, and 20% to the held-out test set:

| Class Name | Train (60%) | Validation (20%) | Test (20%) | Total Count |
| :--- | :---: | :---: | :---: | :---: |
| **Bird-drop** | 120 | 46 | 41 | 207 |
| **Clean** | 115 | 39 | 39 | 193 |
| **Dusty** | 114 | 39 | 37 | 190 |
| **Electrical-damage** | 64 | 18 | 21 | 103 |
| **Physical-damage** | 43 | 13 | 13 | 69 |
| **Snow-Covered** | 72 | 25 | 26 | 123 |
| **Total** | **528 (59.66%)** | **180 (20.34%)** | **177 (20.00%)** | **885 (100.0%)** |

All models were evaluated on the **exact same 177 test samples**.

---

## 2. Model Architecture Comparison

Four model configurations were trained and evaluated under identical experimental conditions (batch size 32, Adam optimizer, cross-entropy loss, CPU environment):

| Metric | MobileNetV2 (Baseline) | EfficientNet-B0 (Production) | EfficientNet-B0 (Augmented) | SparkNet-A (Ablation) |
| :--- | :---: | :---: | :---: | :---: |
| **Test Accuracy** | 83.05% | **85.31%** | 80.23% | 79.66% |
| **Macro Precision** | 86.79% | **87.34%** | 83.21% | 81.45% |
| **Macro Recall** | 82.16% | **83.57%** | 77.89% | 77.20% |
| **Macro F1 Score** | 83.66% | **84.93%** | 78.90% | 78.20% |
| **Weighted F1 Score** | 83.01% | **85.25%** | 79.81% | 79.50% |
| **Total Parameters** | **2,231,558** | 4,015,234 | 4,015,234 | ~525,000 |
| **Model Size** | **8.51 MB** | 15.32 MB | 15.32 MB | 3.83 MB |
| **CPU Latency (ms/img)** | **11.59 ms** | 16.10 ms | 16.15 ms | 14.80 ms |
| **Throughput (FPS)** | **86.3 FPS** | 62.1 FPS | 61.9 FPS | 67.6 FPS |

### Architectural Insights
1. **EfficientNet-B0 vs. MobileNetV2**: EfficientNet-B0 achieved superior overall accuracy (+2.26%) and macro F1 (+1.27%), demonstrating that its compound-scaled depth and squeeze-and-excitation blocks capture finer visual textures in micro-cracks and subtle discolouration. MobileNetV2 offered 28% faster inference and 44% smaller memory footprint, making it a viable alternative for extreme edge devices.
2. **Impact of Data Augmentation (Critical Finding)**: Incorporating domain-specific geometric (rotation $\pm 15^\circ$, flips) and color/contrast augmentations during training **reduced** test accuracy from 85.31% to 80.23% (-5.08%) and macro F1 from 84.93% to 78.90% (-6.03%). In solar panel fault inspection, synthetic distortions can blur the subtle distinction between natural glass reflections and genuine surface defects. Consequently, data augmentation was excluded from the production model.
3. **SparkNet-A**: The reference dual-branch architecture achieved 79.66% accuracy. While highly compact (3.83 MB), its custom convolutional feature extractors underperformed pretrained ImageNet representations.

---

## 3. Production EfficientNet-B0 Per-Class Evaluation

Detailed evaluation metrics for the production EfficientNet-B0 model across the 177 test set samples:

| Class Name | Test Support | Precision | Recall | F1-Score | Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Bird-drop** | 41 | 77.8% | 85.4% | 81.4% | Minor confusion with isolated heavy dust clumps |
| **Clean** | 39 | 85.4% | 89.7% | 87.5% | Strong discrimination against unsoiled reflections |
| **Dusty** | 37 | 81.1% | 81.1% | 81.1% | Balanced precision and recall; minor confusion with faint bird drops |
| **Electrical-damage** | 21 | 90.9% | 95.2% | 93.0% | High sensitivity (20/21 detected); critical for electrical safety |
| **Physical-damage** | 13 | 88.9% | 61.5% | 72.7% | High precision but lower recall due to limited training samples (43 images) |
| **Snow-Covered** | 26 | 100.0% | 88.5% | 93.9% | Zero false positives; perfect precision on opaque snow |
| **Macro Average** | **177** | **87.34%** | **83.57%** | **84.93%** | Unweighted mean across all 6 classes |
| **Weighted Average**| **177** | **85.72%** | **85.31%** | **85.25%** | Sample-weighted mean |

---

## 4. Robustness Testing Under Controlled Perturbations

To evaluate the operational resilience of the production model under real-world sensor and environmental variations, we evaluated 10 controlled perturbation conditions on the 177 test samples:

| Perturbation Condition | Evaluated Accuracy | Macro F1 | Performance Impact | Operational Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Original Benchmark** | 83.62% | 82.35% | Baseline | Unmodified test set baseline |
| **Reduced Resolution** | **61.02%** | **58.59%** | **-22.60%** | **Strongest degradation condition**. Downsampling below $112 \times 112$ destroys fine crack lines and micro-burns. |
| **Gaussian Blur** | **76.84%** | **75.32%** | **-6.78%** | Noticeable degradation; softens high-frequency defect edges and cell boundaries. |
| **Low Light** | **78.53%** | **78.58%** | **-5.09%** | Illumination reduction impairs differentiation between dark bird drops and shadows. |
| **Low Contrast** | 81.36% | 81.39% | -2.26% | Moderate decline; model partially relies on texture gradients. |
| **High Contrast** | 81.92% | 82.49% | -1.70% | Minimal impact; preserved boundary edges maintain classification. |
| **High Brightness** | 82.49% | 82.84% | -1.13% | Minor impact; model demonstrates strong resilience against solar glare. |
| **Small Rotation ($\pm 10^\circ$)** | 83.05% | 83.06% | -0.57% | Negligible change; orientation-tolerant feature representations. |
| **JPEG Compression** | 84.18% | 84.52% | +0.56% | Highly stable; resistant to standard lossy web transmission. |
| **Sensor Noise** | 84.75% | 84.84% | +1.13% | Highly stable; additive Gaussian noise does not disrupt convolutional kernels. |

---

## 5. Evaluation Consistency & Preprocessing Reproducibility

During system benchmarking, an evaluation discrepancy was identified: an early evaluator reported 83.62% test accuracy, while the canonical evaluation script reported 85.31% on the exact same checkpoint (`efficientnet_b0_baseline_best.pth`) and 177 test images.

### Investigation & Root Cause
A forensic comparison (`results/metrics/evaluation_consistency_report.json`) analyzed all 177 image predictions and identified 14 differing classifications:
- **Root Cause**: Resizing interpolation algorithm mismatch.
  - The canonical Phase 4 pipeline utilized OpenCV bilinear interpolation (`cv2.INTER_LINEAR` via Albumentations).
  - The secondary evaluator utilized PIL bicubic interpolation (`Image.Resampling.BICUBIC`).
- **Engineering Resolution**: Because EfficientNet-B0 was trained exclusively on OpenCV bilinear representations, using PIL bicubic introduced subtle sub-pixel antialiasing artifacts that shifted boundary probabilities on 14 borderline images.
- **Outcome**: Standardized `src/preprocessing/pipeline.py` across all evaluation scripts, API inference services, and tests, establishing 85.31% as the canonical, reproducible test benchmark.
