# Methodology

## 1. End-to-End Inspection Pipeline Overview

The system processes solar module imagery through a structured, multi-stage engineering pipeline that transitions from raw optical data ingestion to operational maintenance guidance:

```
[ Input Image (Field Capture / Drone / Upload) ]
                      │
                      ▼
            [ Image Validation ]
    (Format verification, corrupt image check, RGBA -> RGB)
                      │
                      ▼
         [ Canonical Preprocessing ]
       (Bilinear resize 224x224, [0,1] scaling, ImageNet norm)
                      │
                      ▼
         [ EfficientNet-B0 Backbone ]
                      │
                      ▼
     ┌────────────────┴────────────────┐
     ▼                                 ▼
[ ML Outputs ]               [ Explainability ]
• Predicted Class            • Grad-CAM Heatmap (features.8)
• Softmax Confidence                   │
     │                                 ▼
     │                   [ Fault-Region Extraction ]
     │                   • Saliency thresholding (tau = 0.60)
     │                   • Morphological cleaning
     │                   • Bounding box & centroid
     │                   • Visual area percentage (A_percent)
     │                                 │
     └────────────────┬────────────────┘
                      ▼
          [ Visual Severity Triage ]
       (Heuristic composite score -> Tier)
                      │
                      ▼
        [ Maintenance Recommender ]
  (Prescriptive action, urgency, safety checklist)
                      │
                      ▼
        [ Asset Metadata Association ]
    (Panel ID, string/array location, GPS, timestamp)
                      │
                      ▼
     [ Relational Database Persistence ]
         (SQLAlchemy / SQLite / PostgreSQL)
                      │
                      ▼
     [ Next.js 16 User Interface Dashboard ]
```

---

## 2. Distinction of Pipeline Components

To maintain scientific integrity, the system strictly delineates between empirical machine learning predictions, explainability mappings, derived heuristic rules, and operational asset metadata:

| Category | Component | Nature of Output | Source of Ground Truth / Validation |
| :--- | :--- | :--- | :--- |
| **ML Inference** | Predicted Fault Class | Discrete categorical label ($k \in \{0, \dots, 5\}$) | Evaluated against 177 test set labels |
| **ML Inference** | Prediction Confidence | Softmax probability ($p_k \in [0.0, 1.0]$) | Cross-entropy calibrated probability distribution |
| **Explainability** | Grad-CAM Heatmap | Continuous 2D attention map ($L^c \in [0.0, 1.0]^{H \times W}$) | Gradient backpropagation from target class score |
| **Derived Heuristic**| Visual Region Area % | Approximate surface coverage fraction ($A_{\text{roi}} \in [0, 100]$) | Morphological contour extraction over saliency mask |
| **Derived Heuristic**| Visual Severity Grade | Discrete operational risk tier (`CRITICAL` to `NEGLIGIBLE`) | Rule-based matrix (class risk $\times$ confidence $\times$ area) |
| **Derived Heuristic**| Maintenance Advice | Action text, urgency ranking, inspection checklist | Prescriptive domain logic table |
| **Operational Meta** | Panel Asset Data | Panel ID, string location, coordinates, timestamp | Inputted by field technician or system client |

---

## 3. Data Ingestion & Preprocessing

### 3.1 Input Image Validation
Every uploaded image passes through structural integrity checks:
1. **File Format Verification**: Restricts inputs to valid image file types (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
2. **Channel Normalization**:
   - 4-channel images (RGBA) are alpha-composited over a neutral white background:
     $$I_{\text{RGB}} = I_{\text{RGB}} \cdot \alpha + 255 \cdot (1 - \alpha)$$
   - Grayscale images (1 channel) are replicated across three color channels ($C=3$).
3. **Dimension Bounds**: Validates that image dimensions satisfy minimum structural criteria ($H \ge 32, W \ge 32$).

### 3.2 Canonical Preprocessing Pipeline
To guarantee reproducibility across training, testing, and live REST inference, images undergo deterministic transformations in `src/preprocessing/pipeline.py`:
1. **Geometric Standardization**: Bilinear interpolation resizing to $224 \times 224$ pixels via OpenCV (`cv2.INTER_LINEAR`).
2. **Intensity Scaling**: Converts 8-bit unsigned integer channels $[0, 255]$ to 32-bit floating-point values $[0.0, 1.0]$.
3. **Statistical Normalization**: Subtracts ImageNet channel means and scales by standard deviations:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
   $$x_{c, i, j} = \frac{\frac{I_{c, i, j}}{255.0} - \mu_c}{\sigma_c}$$
4. **Channel Reordering**: Transposes memory layout from OpenCV HWC format to PyTorch tensor format $(1, 3, 224, 224)$.

---

## 4. Deep Learning Model Architectures

### 4.1 EfficientNet-B0 (Production Backbone)
EfficientNet-B0 employs compound scaling, uniformly balancing network depth $d$, width $w$, and input resolution $r$ using fixed scaling coefficients:
- **Feature Extractor**: 16 Mobile Inverted Bottleneck Convolution (MBConv) blocks utilizing squeeze-and-excitation optimization and depthwise-separable convolutions.
- **Classification Head**: Replaced terminal 1,000-class head with:
  ```python
  nn.Sequential(
      nn.Dropout(p=0.2),
      nn.Linear(in_features=1280, out_features=6)
  )
  ```
- **Total Parameters**: 4,015,234 (all trainable).

### 4.2 MobileNetV2 (Lightweight Baseline)
MobileNetV2 provides an efficient benchmark designed for edge-device deployment:
- **Feature Extractor**: Inverted residual blocks with linear bottlenecks.
- **Classification Head**: Global average pooling followed by dropout ($p=0.2$) and linear layer ($1280 \to 6$).
- **Total Parameters**: 2,231,558.

### 4.3 Training Methodology
- **Loss Function**: Multi-class cross-entropy loss:
  $$\mathcal{L}_{\text{CE}} = - \sum_{k=0}^{5} y_k \log(\hat{p}_k)$$
- **Optimizer**: Adam ($\text{lr} = 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$).
- **Batch Size & Device**: Batch size 32, executed on CPU hardware.
- **Stopping Criterion**: Checkpoint saved at epoch with lowest validation cross-entropy loss.

---

## 5. Visual Explainability via Grad-CAM

To interpret the decision-making process of EfficientNet-B0, we compute Gradient-weighted Class Activation Mapping (Grad-CAM):
1. **Target Layer**: Select the final convolutional layer of the backbone: `model.features[-1]` (a $1 \times 1$ convolutional projection generating 1,280 feature maps of spatial dimension $7 \times 7$).
2. **Gradient Computation**: Compute the gradient of the unnormalized score for target class $c$ ($y^c$) with respect to feature activation map $A^k$:
   $$\frac{\partial y^c}{\partial A^k_{i, j}}$$
3. **Neuron Importance Weights**: Global-average-pool the gradients over spatial dimensions $(i, j)$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial y^c}{\partial A^k_{i, j}}$$
4. **Weighted Saliency Map**: Perform a rectified linear combination of feature maps:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k=1}^{K} \alpha_k^c A^k\right)$$
5. **Upsampling & Colormap**: Bilinearly upsample $7 \times 7$ saliency map to original image resolution ($224 \times 224$), normalize to $[0.0, 1.0]$, and apply OpenCV `COLORMAP_JET` overlay.

---

## 6. Approximate Visual Fault-Region Extraction

The continuous Grad-CAM heatmap $L^c$ is converted into discrete spatial regions using `src/explainability/fault_region.py`:
1. **Binary Thresholding**: Saliency values exceeding threshold $\tau = 0.60$ (or adaptive Otsu threshold) are activated:
   $$M(i, j) = \begin{cases} 255 & \text{if } L^c(i, j) \ge \tau \\ 0 & \text{otherwise} \end{cases}$$
2. **Morphological Filtering**:
   - Morphological Closing (kernel $5 \times 5$) merges fragmented activations and fills interior holes.
   - Morphological Opening (kernel $3 \times 3$) eliminates spurious isolated pixel noise.
3. **Contour Extraction**: Identifies external connected contours using `cv2.findContours`. Contours with area smaller than 25 pixels are filtered out.
4. **Spatial Metrics Extraction**:
   - **Bounding Box**: Rectangular coordinates $(x, y, w, h)$.
   - **Centroid**: Center of mass $(\bar{x}, \bar{y})$ computed from image spatial moments ($m_{10}/m_{00}, m_{01}/m_{00}$).
   - **Visual Area Coverage Percentage**:
     $$A_{\text{percent}} = \frac{\sum_{i, j} \mathbb{I}[M(i, j) = 255]}{H \times W} \times 100\%$$

> [!NOTE]
> The extracted bounding box indicates the center of visual attention and does not constitute a certified CAD boundary of structural defect geometry.

---

## 7. AI-Assisted Visual Severity Estimation

Severity is computed using a multi-factor rule-based heuristic (`src/severity/severity_estimator.py`):
1. **Class Inherent Risk Weight ($W_c$)**:
   - `Clean`: 0.00
   - `Dusty`: 0.55
   - `Bird-drop`: 0.70
   - `Snow-Covered`: 0.75
   - `Physical-damage`: 0.90
   - `Electrical-damage`: 1.00
2. **Composite Severity Score**:
   $$S = W_c \times \left(0.5 \times p_c + 0.5 \times \min\left(1.0, \frac{A_{\text{percent}}}{A_{\text{norm}}}\right)\right)$$
   where $A_{\text{norm}}$ is $20.0\%$ for localized faults (`Bird-drop`, `Electrical-damage`, `Physical-damage`) and $40.0\%$ for diffuse surface faults (`Dusty`, `Snow-Covered`).
3. **Tier Classification**:
   - `CRITICAL`: $S \ge 0.75$ or `Electrical-damage` with $A_{\text{percent}} \ge 10.0\%$
   - `HIGH`: $0.55 \le S < 0.75$
   - `MEDIUM`: $0.35 \le S < 0.55$
   - `LOW`: $0.15 \le S < 0.35$
   - `NEGLIGIBLE`: $S < 0.15$ or `Clean`
4. **Low-Confidence Override**: If top-1 confidence $p_c < 0.60$, a confidence warning is appended, and `manual_inspection_recommended` is forced to `True`.

---

## 8. Prescriptive Maintenance Recommendation

The maintenance engine (`src/maintenance/maintenance_recommender.py`) translates the diagnostic tuple $(\text{Class}, p_c, \text{Severity}, A_{\text{percent}})$ into standardized operational workflows:

| Predicted Class | Severity Tier | Operational Urgency | Recommended Action |
| :--- | :--- | :--- | :--- |
| **Electrical-damage** | Any | `IMMEDIATE_REVIEW` | Disconnect panel/string circuit; dispatch licensed electrical technician for bypass diode, cell hotspot, and junction box testing. |
| **Physical-damage** | `HIGH` / `CRITICAL` | `PRIORITY` | Perform on-site structural audit; inspect for glass shattering, moisture ingress, and ground fault hazards; evaluate module replacement. |
| **Physical-damage** | `MEDIUM` / `LOW` | `SCHEDULED` | Log micro-crack location; monitor string IV curve for performance degradation. |
| **Bird-drop** | `HIGH` | `PRIORITY` | Schedule localized high-pressure or deionized water cleaning within 48 hours to avert localized reverse-bias hot-spot formation. |
| **Bird-drop** | `LOW` / `MEDIUM` | `SCHEDULED` | Include in upcoming routine array washing cycle. |
| **Dusty** | `HIGH` | `SCHEDULED` | Schedule comprehensive array cleaning; evaluate local environmental soiling rates. |
| **Dusty** | `LOW` / `MEDIUM` | `ROUTINE` | Monitor daily string energy yield against clean baseline arrays. |
| **Snow-Covered** | `HIGH` | `PRIORITY` | Inspect for snow shedding obstruction and mechanical rack loading; evaluate gentle manual clearing if ambient temperatures remain below freezing. |
| **Clean** | `NEGLIGIBLE` | `ROUTINE` | Optimal operational state; continue routine automated monitoring. |

---

## 9. Relational Data Persistence & Metadata

All inspection records and panel assets are managed through SQLAlchemy models:
- **`Panel` Entity**: Stores unique `panel_id`, string/array identifier, geographic coordinates (latitude, longitude), installation date, and rated power (W).
- **`Inspection` Entity**: Stores foreign key to `Panel`, inspection timestamp, original image filename, predicted class, confidence, visual region metrics $(x, y, w, h, A_{\text{percent}})$, severity tier, urgency, maintenance recommendation text, and technician review flags.
- **Engine Abstraction**: Configured with SQLite (`data/solar_panel_ai.db`) for lightweight local execution, with zero-code migration compatibility for PostgreSQL via environment variables.
