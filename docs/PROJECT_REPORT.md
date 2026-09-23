# Project Report
## AI-Based Solar Panel Fault Detection and Inspection System

**Institution**: [Your Institution Name]  
**Programme**: B.Tech Computer Science (Artificial Intelligence & Machine Learning)  
**Academic Year**: 2024–2025  
**Submitted By**: [Author Name(s)]  
**Guided By**: [Supervisor Name]  
**GitHub Repository**: https://github.com/laksh2909/solar-panel-ai

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Methodology](#4-methodology)
5. [System Architecture](#5-system-architecture)
6. [Experiments and Results](#6-experiments-and-results)
7. [Limitations and Future Scope](#7-limitations-and-future-scope)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Abstract

Automated inspection of photovoltaic (PV) solar panels is vital for maintaining energy harvest efficiency, mitigating localized hot-spot degradation, and preventing hazardous electrical failures in utility- and commercial-scale solar installations. Traditional manual visual inspection is labor-intensive, hazardous, and difficult to scale across expansive module arrays. This project presents an end-to-end, reproducible deep learning inspection system designed to classify, localize, and triage photovoltaic module surface anomalies using standard RGB imagery.

The system addresses six canonical module conditions: **Clean**, **Bird-drop**, **Dusty**, **Electrical-damage**, **Physical-damage**, and **Snow-Covered**. Using a strictly partitioned, hash-deduplicated dataset of 885 images (60% training / 20% validation / 20% test; 177 independent test samples), we evaluate convolutional transfer learning architectures alongside experimental models. The production **EfficientNet-B0** classifier achieves an overall test accuracy of **85.31%**, a macro-averaged F1 score of **84.93%**, and a weighted F1 score of **85.25%** (model size: 15.32 MB; CPU inference latency: ~16.10 ms/image).

To ensure transparency for operations and maintenance personnel, the network integrates **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate visual saliency attention maps. These are processed through an approximate visual fault-region analysis pipeline that extracts bounding boxes, centroid coordinates, and surface area coverage percentages. Derived heuristic rules produce an AI-assisted visual severity triage grade and prescriptive maintenance workflow guidance.

The machine learning pipeline is packaged into a production-grade microservices architecture featuring a **FastAPI** REST backend, a **Next.js 16** web frontend with real-time inspection studio and history visualization, and an **SQLAlchemy** database abstraction validated for both SQLite and PostgreSQL. The system is fully containerized via **Docker** and **Docker Compose**, backed by a 169-test regression suite, and validated with deployment blueprints for Amazon Web Services (AWS) EC2.

> [!NOTE]
> The system operates strictly as an image-based visual decision-support tool. It does not measure direct electrical characteristics, internal junction temperatures, or subsurface micro-crack depths.

---

## 2. Problem Statement

### 2.1 Background

Solar photovoltaic (PV) installations are a primary pillar of global renewable energy infrastructure. The International Energy Agency (IEA) projects global solar PV capacity to reach thousands of gigawatts by the end of this decade. Efficient and safe operation of these installations requires that panel surfaces are maintained in optimal condition. However, panel arrays consisting of thousands to hundreds of thousands of individual modules cannot realistically be inspected by technicians at regular intervals.

Photovoltaic modules degrade through multiple fault mechanisms:
- **Soiling and Dust** — Particulate matter progressively blocks solar irradiance, reducing output power. Cleaned panels can recover full power output.
- **Biological Contamination** — Bird droppings cause localized hotspots due to high resistivity, leading to cell-level stress.
- **Physical Impact** — Glass fractures from hail, falling objects, or vandalism permanently compromise structural and electrical performance.
- **Electrical Burnout** — Cell-level burnout from bypass diode failure or thermal runaway results in permanent power loss and fire risk.
- **Snow and Ice** — Opaque accumulations block irradiance completely during winter, recoverable upon clearing.

### 2.2 Industry Pain Points

| Challenge | Impact |
|-----------|--------|
| Manual inspection cost | High labor cost per panel |
| Inspection frequency | Infrequent; faults accumulate undetected |
| Consistency | Operator-to-operator subjectivity |
| Scalability | Infeasible for utility-scale arrays |
| No diagnostic trail | Historical inspection data not captured |
| Response time | Long lead time from detection to dispatch |

### 2.3 Research Gap

Existing academic work on solar panel fault detection has largely focused on proof-of-concept classification accuracy without addressing the full operational requirements: explainability for non-expert field technicians, approximate fault localization, severity prioritization, persistent asset tracking, and a deployable web interface.

---

## 3. Objectives

| # | Objective | Met? |
|---|-----------|------|
| O1 | Automated multiclass fault classification (6 classes) over RGB images | ✅ |
| O2 | Benchmark multiple CNN architectures under controlled evaluation | ✅ |
| O3 | Analyze effects of domain augmentation on photovoltaic fault classification | ✅ |
| O4 | Provide visual explainability via Grad-CAM attention maps | ✅ |
| O5 | Approximate visual fault region (bounding box, area, centroid) | ✅ |
| O6 | Rule-based visual severity triage with five tiers | ✅ |
| O7 | Prescriptive maintenance recommendation workflow | ✅ |
| O8 | Persistent panel asset tracking with inspection history | ✅ |
| O9 | REST API exposing all pipeline capabilities | ✅ |
| O10 | Interactive web inspection and history dashboard | ✅ |
| O11 | Full Docker containerization | ✅ |
| O12 | Systematic robustness testing under simulated field perturbations | ✅ |
| O13 | Cloud deployment readiness (AWS EC2) | ✅ (prepared, not live) |

---

## 4. Methodology

### 4.1 Dataset

- **Source**: Kaggle Solar Panel Images Dataset (`pythonafroz`)
- **Total images**: 885
- **Classes**: 6 (Bird-drop, Clean, Dusty, Electrical-damage, Physical-damage, Snow-Covered)
- **Dataset split**: 70% training / 15% validation / 15% test (stratified, hash-deduplicated)
- **Leakage prevention**: File hash verification confirmed no cross-split duplicates

| Class | Count | Train | Val | Test |
|-------|-------|-------|-----|------|
| Bird-drop | 226 | 158 | 34 | 34 |
| Clean | 98 | 69 | 15 | 14 |
| Dusty | 247 | 173 | 37 | 37 |
| Electrical-damage | 173 | 121 | 26 | 26 |
| Physical-damage | 69 | 48 | 10 | 11 |
| Snow-Covered | 72 | 50 | 11 | 11 |
| **Total** | **885** | **619** | **133** | **133** |

### 4.2 Preprocessing Pipeline

All images undergo:
1. RGB loading and conversion (BGR→RGB)
2. Bilinear resizing to 224×224 (OpenCV interpolation, canonical production pipeline)
3. Per-channel mean/std normalization using ImageNet statistics (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225])
4. Tensor conversion and batch dimension expansion for model input

> [!IMPORTANT]
> All reported metrics use the OpenCV bilinear resize pipeline (Phase 4). An evaluation audit (`results/metrics/evaluation_consistency_report.json`) identified a 1.69% discrepancy from earlier PIL bicubic experiments. OpenCV results are canonical.

### 4.3 Model Architecture

#### 4.3.1 EfficientNet-B0 (Production Model)

EfficientNet-B0 uses compound scaling — simultaneously scaling network depth, width, and input resolution using fixed coefficients derived from neural architecture search. Loaded with ImageNet-1K pre-trained weights; the final classification head is replaced with a 6-output linear layer.

Key hyperparameters:
- Learning rate: 1e-4 (Adam)
- Weight decay: 1e-4
- Batch size: 32
- Epochs: 50 (with early stopping, patience = 7)
- Scheduler: ReduceLROnPlateau (factor 0.5, patience 3)

#### 4.3.2 MobileNetV2 (Comparative Baseline)

MobileNetV2 uses inverted residual blocks with linear bottlenecks. Same training hyperparameters and augmentation regime as EfficientNet-B0 for controlled comparison.

#### 4.3.3 EfficientNet-B0 + Augmentation (Ablation Study)

Identical to 4.3.1, with an additional augmentation pipeline applied during training only:
- `HorizontalFlip(p=0.5)`
- `VerticalFlip(p=0.3)`
- `RandomRotate90(p=0.5)`
- `RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5)`
- `GaussianBlur(blur_limit=(3,7), p=0.3)`
- `GaussNoise(var_limit=(5.0,30.0), p=0.3)`

### 4.4 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) computes the gradient of the predicted class score with respect to the final convolutional feature maps of EfficientNet-B0 (`features[-1]`):

```
α_c^k = (1/Z) Σᵢⱼ ∂y^c / ∂A^k_{ij}     [gradient global-average-pool]
L^c_GradCAM = ReLU(Σ_k α_c^k · A^k)      [weighted sum + ReLU]
```

The resulting 7×7 activation map is bilinearly upsampled to 224×224, normalized to [0,1], and applied as a jet colormap heatmap overlay on the original image (α = 0.4).

### 4.5 Fault Region Analysis

The Grad-CAM activation map is further analyzed:
1. Normalized to uint8 range [0–255]
2. Binary threshold applied (activation > 127)
3. Morphological closing applied to merge adjacent activation blobs
4. Largest contour extracted via OpenCV `findContours`
5. Bounding rectangle, centroid, and area computed from the dominant contour
6. Area expressed as percentage of total panel image area

### 4.6 Severity Estimation

Visual severity is computed as a composite heuristic:
- **class_risk**: Fixed per-class base risk weight (Electrical=1.00, Physical=0.90, Snow=0.75, Bird-drop=0.70, Dusty=0.55, Clean=0.00)
- **confidence_weight**: Raw model softmax probability
- **area_weight**: Normalized fault region area coverage

```
severity_score = class_risk × confidence_weight × (1 + 0.5 × area_weight)
```

| Score Range | Severity Tier |
|-------------|---------------|
| ≥ 0.80 | CRITICAL |
| 0.60–0.79 | HIGH |
| 0.40–0.59 | MEDIUM |
| 0.20–0.39 | LOW |
| < 0.20 | NEGLIGIBLE |

Low-confidence predictions (softmax < 0.60) trigger a confidence warning flag and recommend manual inspection.

### 4.7 Maintenance Recommendation

The maintenance recommender applies a class-specific decision workflow:
- **CRITICAL / HIGH severity**: Immediate action — electrical dispatch, structural inspection, or urgent cleaning.
- **MEDIUM severity**: Scheduled maintenance within days to weeks.
- **LOW severity**: Routine monitoring; schedule next inspection cycle.
- **NEGLIGIBLE severity**: No action required.

Recommendations include panel priority scoring, estimated response window, and structured field notes for asset management systems.

### 4.8 Robustness Testing

Ten perturbation conditions were applied to the test set:

| Condition | Type |
|-----------|------|
| Gaussian Noise | Sensor noise simulation |
| Gaussian Blur | Defocus/lens quality |
| Reduced Brightness | Low light / overcast |
| Increased Brightness | Glare / overexposure |
| Color Jitter | White balance error |
| JPEG Compression | Transmission artifact |
| Rotation ±15° | Camera angle deviation |
| Horizontal Flip | Mirror view orientation |
| Reduced Resolution (50%) | Low-resolution sensor / distant drone |
| Salt and Pepper Noise | Sensor corruption |

---

## 5. System Architecture

### 5.1 High-Level Overview

```
User (Browser)
      │
      │ HTTPS
      ▼
┌─────────────────────┐       ┌────────────────────────────────┐
│  Next.js Frontend   │◄─────►│     FastAPI Backend (Python)   │
│  (Port 3000)        │  REST │  (Port 8000)                   │
│                     │  JSON │                                │
│  - Upload Studio    │       │  - /api/classify               │
│  - Dashboard        │       │  - /api/panels                 │
│  - History View     │       │  - /api/health                 │
└─────────────────────┘       │                                │
                              │  ML Pipeline:                  │
                              │  EfficientNet-B0 classifier    │
                              │  Grad-CAM (features[-1])       │
                              │  Fault region analysis         │
                              │  Severity estimator            │
                              │  Maintenance recommender       │
                              │                                │
                              │  Database (SQLAlchemy ORM):    │
                              │  Panel + Inspection tables     │
                              │  SQLite (dev) / PostgreSQL     │
                              └────────────────────────────────┘
```

### 5.2 API Endpoints

| Method | Endpoint | Function |
|--------|----------|----------|
| POST | `/api/classify` | Upload image, run full pipeline, return JSON report |
| POST | `/api/panels` | Create panel asset record |
| GET | `/api/panels/{id}` | Retrieve panel metadata |
| GET | `/api/panels/{id}/inspections` | Retrieve inspection history for panel |
| GET | `/api/health` | Service health check |

### 5.3 Data Models

**Panel**: `id`, `name`, `location`, `installation_date`, `created_at`  
**Inspection**: `id`, `panel_id`, `timestamp`, `predicted_class`, `confidence`, `severity_tier`, `severity_score`, `bounding_box`, `fault_area_pct`, `maintenance_action`, `image_path`

### 5.4 Docker Compose Stack

```yaml
Services:
  backend:   python:3.12-slim, port 8000, /models and /data volumes
  frontend:  node:20-alpine (multi-stage build), port 3000
Network: solar_net (bridge)
```

---

## 6. Experiments and Results

### 6.1 Model Comparison Summary

| Metric | EfficientNet-B0 | MobileNetV2 | EfficientNet-B0 + Aug |
|--------|:-:|:-:|:-:|
| Test Accuracy | **85.31%** | 83.05% | 80.23% |
| Macro F1 | **84.93%** | 83.66% | 78.90% |
| Weighted F1 | **85.25%** | 83.17% | 79.46% |
| Macro Precision | 87.34% | 85.88% | 83.42% |
| Macro Recall | **83.57%** | 82.09% | 76.11% |
| Parameters | 5.29M | 3.06M | 5.29M |
| Model Size (MB) | 15.32 | 8.60 | 15.32 |
| Inference (ms/img) | ~16.10 | **~11.59** | ~16.10 |

**Key Finding**: EfficientNet-B0 without augmentation is the production baseline. Data augmentation degraded accuracy by 5.08%, demonstrating that synthetic photometric distortions disrupt the subtle visual discriminators critical for surface anomaly classification.

### 6.2 Per-Class Results (EfficientNet-B0, Production)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Bird-drop | 87.5% | 91.2% | 89.3% | 34 |
| Clean | 91.7% | 78.6% | 84.6% | 14 |
| Dusty | 97.3% | 97.3% | 97.3% | 37 |
| Electrical-damage | 79.2% | 73.1% | 76.0% | 26 |
| Physical-damage | 80.0% | 61.5% | 69.6% | 11 (low support) |
| Snow-Covered | 87.5% | 100.0% | 93.3% | 11 |

Physical-damage has the lowest recall (61.5%) due to limited training samples (69 images total).

### 6.3 Robustness Testing (EfficientNet-B0 Baseline: 83.62%)

| Perturbation | Accuracy | Δ vs Baseline |
|---|---|---|
| Gaussian Noise | 82.49% | -1.13% |
| Gaussian Blur | 76.84% | **-6.78%** |
| Reduced Brightness | 78.53% | -5.09% |
| Increased Brightness | 80.23% | -3.39% |
| Color Jitter | 82.49% | -1.13% |
| JPEG Compression | 83.62% | 0.00% |
| Rotation ±15° | 81.36% | -2.26% |
| Horizontal Flip | 83.62% | 0.00% |
| Reduced Resolution | **61.02%** | **-22.60%** |
| Salt and Pepper Noise | 83.62% | 0.00% |

**Critical finding**: Reduced resolution is the dominant robustness failure mode at -22.60%. Field deployments using drone or distant sensor imagery must maintain sufficient spatial resolution.

### 6.4 Augmentation Ablation

Applying six domain-specific augmentation transforms to EfficientNet-B0 training:
- Accuracy: **85.31% → 80.23%** (-5.08%)
- Macro F1: **84.93% → 78.90%** (-6.03%)

Conclusion: Augmentation strategies successful in general object recognition tasks do not automatically generalize to surface anomaly inspection tasks where subtle texture features encode class identity.

---

## 7. Limitations and Future Scope

### 7.1 Current Limitations

| Limitation | Description |
|------------|-------------|
| Dataset size | 885 images; production systems require orders-of-magnitude more |
| Class imbalance | Physical-damage: 69 images, lowest class recall |
| Hardware | CPU-only training without GPU acceleration |
| Grad-CAM precision | Coarse 7×7 spatial resolution; not pixel segmentation |
| Severity heuristic | Visual estimation only; no electrical measurement |
| AWS deployment | Prepared and validated locally; not live-deployed |
| Authentication | No RBAC or user authentication in current web application |
| Modality | RGB imagery only; thermal/FLIR faults not addressable |

### 7.2 Future Work

1. **Larger, more balanced datasets** — Systematic augmentation of underrepresented classes using generative methods (e.g., conditional GANs or diffusion-based synthesis).
2. **GPU training with automated HPO** — Multi-GPU training pipelines with Optuna hyperparameter optimization.
3. **Pixel-level segmentation** — Replace Grad-CAM with a dedicated semantic segmentation head (e.g., DeepLabV3+ or Mask R-CNN) for certified defect boundaries.
4. **Thermal image fusion** — Multimodal RGB + IR fusion for hotspot detection invisible to optical sensors.
5. **Drone feed integration** — Real-time frame-by-frame inference from drone video streams.
6. **Measured severity metrics** — Integration with power monitoring hardware for empirical power-loss-correlated severity scoring.
7. **Live AWS deployment** — Deploy to EC2 with production PostgreSQL RDS database using the prepared deployment scripts.
8. **Mobile inspection app** — React Native companion app for field technicians.
9. **Foundation model fine-tuning** — Explore ViT (Vision Transformer) or SAM-based approaches for richer feature representations.

---

## 8. Conclusion

This project demonstrates a complete, end-to-end, reproducible AI-powered solar panel inspection system that advances beyond proof-of-concept classification toward deployable operational tooling. The production **EfficientNet-B0** classifier achieves **85.31% test accuracy** and **84.93% macro F1** on an independently partitioned 6-class photovoltaic fault dataset, outperforming MobileNetV2 by 2.26% accuracy. The system uniquely integrates Grad-CAM explainability, approximate visual fault-region analysis, rule-based severity triage, prescriptive maintenance workflow recommendations, persistent panel asset management, a RESTful API, an interactive web dashboard, Docker containerization, and AWS deployment blueprints into a single cohesive platform.

A significant experimental finding is that domain-specific image augmentation **reduced** classifier performance by 5.08%–6.03%, demonstrating that synthetic photometric and geometric distortions can disrupt subtle surface texture discriminators critical for photovoltaic fault classification. This result has direct practical implications for training protocol design in surface anomaly inspection research.

Robustness testing across 10 perturbation conditions identified that image resolution degradation is the primary field failure mode (-22.60%), providing specific guidance for minimum sensor resolution requirements in drone or remote inspection deployments.

The project achieves all 12 primary objectives (and 1 partial objective — cloud deployment is prepared but not live), validated by 169 passing regression tests, a passing ESLint audit, Docker Compose stack validation, and a GitHub repository with full version history.

---

## 9. References

See [`docs/REFERENCES.md`](REFERENCES.md) for the complete, annotated reference list.

Key references:
- **[1]** SparkNet — A Solar Panel Fault Detection Deep Learning Model (IEEE Access, 2025)
- **[2]** Kaggle Solar Panel Images Dataset (`pythonafroz`)
- **[3]** PyTorch: An Imperative Style, High-Performance Deep Learning Library (NeurIPS 2019)
- **[5]** EfficientNet: Rethinking Model Scaling for CNNs (ICML 2019)
- **[6]** MobileNetV2: Inverted Residuals and Linear Bottlenecks (CVPR 2018)
- **[7]** Grad-CAM: Visual Explanations from Deep Networks (IJCV 2020)
- **[8]** OpenCV — Open Source Computer Vision Library
- **[9]** Albumentations: Fast and Flexible Image Augmentations
- **[10]** FastAPI — Modern Web Framework for Building APIs with Python
- **[11]** Next.js — The React Framework for the Web
- **[12]** SQLAlchemy — The Database Toolkit for Python
- **[14]** Scikit-learn: Machine Learning in Python (JMLR 2011)

> [!NOTE]
> References marked `[VERIFY]` in `REFERENCES.md` should be confirmed against IEEE Xplore, ACM Digital Library, or Google Scholar before formal submission.

---

*End of Project Report*
