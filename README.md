# AI-Based Solar Panel Fault Detection and Inspection System

An end-to-end, production-grade deep learning and computer vision inspection system for automated photovoltaic (PV) defect classification, visual attention explainability (Grad-CAM), severity triage, maintenance workflow guidance, and panel metadata management.

---

## 1. Project Overview

The **Solar Panel Fault Detection and Inspection System** automates the identification and operational triage of defects, contamination, and environmental degradation on photovoltaic modules. Built upon empirical transfer learning architectures (EfficientNet-B0 and MobileNetV2), the system pairs high-performance neural classification with visual explainability, spatial fault localization heuristics, rule-based operational triage, and a modern microservices web interface (FastAPI backend + Next.js 16 frontend + relational database persistence).

The system is containerized with Docker and Docker Compose, features a comprehensive 169-test automated regression suite, and includes local validation configurations for cloud/EC2 deployment.

---

## 2. Problem Statement

Manual visual inspection of commercial and utility-scale solar arrays is labor-intensive, hazardous, and difficult to scale across thousands of distributed modules. Dust accumulation, bird droppings, snow obstruction, micro-cracks, and internal electrical faults degrade photovoltaic generation efficiency and can cause localized hot-spots that lead to permanent panel failure or fire hazards. 

This project provides an automated, reproducible inspection pipeline capable of ingesting high-resolution RGB imagery from field technicians or unmanned aerial vehicles (UAVs/drones), classifying anomalies with calibrated confidence, localizing affected visual regions, estimating operational severity, and generating actionable maintenance recommendations.

---

## 3. Key Features

- **Transfer Learning Classifier**: PyTorch-based EfficientNet-B0 production backbone (85.31% test accuracy, 84.93% macro F1).
- **Comparative Benchmarking**: Evaluated against MobileNetV2 baseline and domain-augmented variants.
- **Visual Saliency (Grad-CAM)**: Heatmap attention overlays isolating discriminative module features.
- **Approximate Fault-Region Analysis**: Morphological contour detection extracting bounding boxes, centroid coordinates, and surface area coverage percentage.
- **Rule-Based Severity Triage**: AI-assisted multi-factor severity estimation (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `NEGLIGIBLE`).
- **Operational Maintenance Engine**: Prescriptive repair, cleaning, and safety action workflows with priority rankings.
- **Asset Metadata Tracking**: Panel ID, location/string tags, latitude/longitude, array name, and operational status.
- **RESTful API Backend**: FastAPI service exposing validation, inference, image uploads, batch inspection, and CRUD history endpoints with OpenAPI/Swagger docs.
- **Interactive Web Interface**: Next.js 16 frontend with TailwindCSS, dynamic dashboards, file dropzones, real-time inspection visualization, and historical filtering.
- **Hybrid Storage Layer**: SQLAlchemy database abstraction supporting local SQLite persistence with seamless PostgreSQL compatibility.
- **Dockerized Architecture**: Multi-stage production containerization with independent backend and frontend services.
- **169 Automated Tests**: Comprehensive regression suite covering preprocessing, models, explainability, severity, API endpoints, database persistence, and container configs.

---

## 4. System Architecture

```
                                  +-------------------------------------------------+
                                  |              Client Web Browser                 |
                                  |     (http://localhost:3000 -> Next.js UI)       |
                                  +-------------------------------------------------+
                                           |                                |
                                HTTP Page  |                                | Direct REST API Calls
                                Navigation |                                | (Uploads, Predictions, CRUD)
                                           v                                v
                        +----------------------+                +----------------------+
                        |   Next.js Frontend   |                |   FastAPI Backend    |
                        |     (Port 3000)      |                |     (Port 8000)      |
                        +----------------------+                +----------------------+
                                                                            |
                                           +--------------------------------+--------------------------------+
                                           |                                |                                |
                                           v                                v                                v
                                +--------------------+           +--------------------+           +--------------------+
                                |   PyTorch Engine   |           |   Explainability   |           |  Database Storage  |
                                |  (EfficientNet-B0) |           |  (Grad-CAM & RoI)  |           | (SQLite / Postgres)|
                                +--------------------+           +--------------------+           +--------------------+
```

---

## 5. ML Pipeline

The machine learning pipeline guarantees deterministic data processing from raw image ingestion to model output:

1. **Robust Loading & Format Normalization**: Ingests JPEG, PNG, WEBP, and BMP images; handles RGBA alpha composite flattening and grayscale replication to ensure 3-channel RGB.
2. **Standardized Resizing**: Bilinear interpolation scaling to canonical $224 \times 224$ pixels.
3. **Intensity Scaling & ImageNet Normalization**: Scaled to $[0.0, 1.0]$, followed by normalization with ImageNet parameters ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
4. **PyTorch Tensor Construction**: Transposed to channel-first $(C, H, W)$ float32 tensor.
5. **Feedforward Classification**: EfficientNet-B0 convolutional feature extraction and linear classification head.
6. **Softmax Probabilities & Uncertainty Auditing**: Calibration check issuing low-confidence warnings when top-1 prediction probability falls below 0.60.

---

## 6. Supported Fault Classes

The model classifies PV module conditions across six canonical operational states:

| Class Name | Visual Description | Primary Operational Concern |
| :--- | :--- | :--- |
| **Bird-drop** | Concentrated organic deposit patches | Severe localized shading, hotspot generation, reverse-bias cell heating |
| **Clean** | Clear, unobstructed glass surface | Baseline normal state; optimal energy conversion |
| **Dusty** | Uniform or streaked particulate layer | Diffuse irradiance reduction, linear power yield loss |
| **Electrical-damage** | Burn marks, cell discolouration, solder melting | Internal cell failure, bypass diode failure, catastrophic fire risk |
| **Physical-damage** | Glass fractures, impact cracks, delamination | Moisture ingress, ground faults, accelerated structural degradation |
| **Snow-Covered** | Heavy opaque snow/ice sheet occlusion | Total circuit blackout, unbalanced string voltages, mechanical load stress |

---

## 7. Explainability

Interpretability is implemented via **Grad-CAM** (Gradient-weighted Class Activation Mapping):
- Targets the terminal convolutional layer (`features.8` in EfficientNet-B0) where high-level semantic feature representations are richest.
- Backpropagates class gradients to compute weight coefficients for each feature map.
- Produces normalized spatial heatmaps identifying which visual regions drove the network's prediction.

> [!NOTE]
> **Technical Scope**: Grad-CAM heatmaps provide approximate visual attention and saliency indication. They do **not** represent pixel-perfect semantic defect segmentation or CAD-level fault boundaries.

---

## 8. Visual Fault-Region Analysis

To translate continuous Grad-CAM heatmaps into structured spatial measurements, the system executes an automated computer vision contour analysis pipeline:
- **Thresholding**: Otsu and percentile-based thresholding identify primary activation hotspots.
- **Morphological Filtering**: Morphological closing fills internal gaps and removes peripheral sensor noise.
- **Contour Extraction**: Identifies the primary connected defect region.
- **Spatial Metrics**: Calculates the bounding box coordinates $(x, y, w, h)$, region centroid, and estimated percentage of module surface area affected.

---

## 9. Severity Estimation

The inspection engine computes an operational severity grade (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `NEGLIGIBLE`) by combining:
1. **Predicted Fault Category**: Inherent operational risk associated with the defect class (e.g., `Electrical-damage` carries higher risk weight than `Dusty`).
2. **Prediction Confidence**: Model certainty in the classification.
3. **Visual Region Extent**: Estimated surface area coverage derived from fault-region analysis.

> [!IMPORTANT]
> **Engineering Scope**: Severity ratings are AI-assisted visual heuristics designed for triage prioritization. They do **not** directly measure electrical power degradation (kW), internal junction temperature ($^\circ\text{C}$), crack micro-depth ($\mu\text{m}$), or physical structural integrity.

---

## 10. Maintenance Recommendation

Based on the predicted class, severity tier, and visual area, the system outputs actionable maintenance workflows:
- **Immediate Dispatch**: Required for electrical faults and severe physical damage posing electrical arcing or safety risks.
- **Scheduled Cleaning**: Recommended for bird contamination and moderate dust accumulation to prevent long-term hotspot formation.
- **Seasonal Monitoring**: Recommended for snow coverage, tracking ambient thaw conditions before manual clearing.
- **Inspection Checklist**: Flags whether manual on-site technician verification or aerial drone re-scanning is advised.

> [!NOTE]
> **Advisory Scope**: Maintenance suggestions constitute operational decision-support guidance and do not replace certified engineering diagnoses or site-specific electrical safety protocols.

---

## 11. Panel Metadata

To support asset tracking in commercial installations, the system records structured solar panel metadata:
- **Panel Identifier**: Unique barcode/string ID (e.g., `SP-FIELD-A-104`).
- **Location / String**: Physical position or array group (e.g., `Rooftop Array North - String 3`).
- **Coordinates**: Latitude and longitude coordinates for geospatial mapping.
- **Installation Date & Rated Capacity**: Panel wattage and commissioning date.
- **Inspection History**: Persistent ledger of all historical inspections, classifications, and severity changes.

---

## 12. Backend

Built with **FastAPI** (Python 3.12):
- **Endpoints**:
  - `POST /api/inspect`: Ingests image file and panel metadata; returns prediction, Grad-CAM heatmap, RoI metrics, severity, and maintenance advice.
  - `GET /api/inspections`: Paginated inspection history with class and date filtering.
  - `GET /api/inspections/{id}`: Detailed inspection report by ID.
  - `GET /api/panels`: List registered panel assets.
  - `GET /api/health`: Service health, model status, and device probe.
- **Validation**: Pydantic schemas enforcing input constraints.
- **CORS**: Configured for browser client communication.
- **Documentation**: Interactive OpenAPI docs available at `/docs` and `/redoc`.

---

## 13. Frontend

Built with **Next.js 16** (React 19, TypeScript, TailwindCSS):
- **Responsive Dashboard**: Summary statistics, operational health metrics, and defect distribution charts.
- **Interactive Inspection Studio**: Drag-and-drop image upload, panel ID assignment, and real-time inference triggering.
- **Diagnostic Result Viewer**: Side-by-side comparison of original image and Grad-CAM heatmap overlay, severity badge, and maintenance action cards.
- **Inspection Ledger**: Searchable, filterable historical record table with export capabilities.
- **Asset Detail Pages**: Deep dive into individual panel inspection histories.

---

## 14. Database

- **ORM**: SQLAlchemy with declarative models (`Panel`, `Inspection`).
- **Default Storage**: SQLite (`data/solar_panel_ai.db`) for lightweight, zero-configuration local and containerized development.
- **Enterprise Storage**: Fully compatible with PostgreSQL via connection string configuration:
  ```bash
  DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/solar_panel_ai
  ```
- **Seeding & Migrations**: Automated database initialization and demo data seeding scripts provided in `scripts/setup_database.py`.

---

## 15. Docker

Production multi-service container deployment via Docker Compose:
- **Backend Service**: Multi-stage Python 3.12 slim image with PyTorch CPU optimization and OpenCV headless dependencies.
- **Frontend Service**: Multi-stage Node.js 20 Alpine image running standalone Next.js production server.
- **Data Volume**: Persistent volume mapping `./data:/app/data` ensuring SQLite state survives container lifecycles.
- **Container Healthchecks**: Built-in health probes on both services (`curl` on backend, `wget spider` on frontend).

---

## 16. AWS Deployment Status

> [!IMPORTANT]
> **Current Cloud Status**: EC2 deployment configuration was prepared and locally validated. Live AWS deployment was not performed because AWS credentials/EC2 access were unavailable and the project intentionally avoids cloud costs at this stage.

Full deployment artifacts (Ubuntu 24.04 provisioning script, systemd service templates, nginx reverse proxy blueprints, and EC2-specific Docker Compose configs) are maintained and documented in [`AWS.md`](AWS.md).

---

## 17. Model Comparison

Four architectures and experimental configurations were rigorously trained and evaluated on an identical, strictly partitioned test set (177 images, zero data leakage):

| Model Architecture | Augmentation Pipeline | Parameters | Checkpoint Size | CPU Latency (ms) | Test Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0 (Production)** | None (Deterministic) | 4,015,234 | 15.32 MB | 16.10 ms | **85.31%** | **84.93%** | **85.25%** |
| **MobileNetV2 (Baseline)** | None (Deterministic) | 2,231,558 | 8.51 MB | 11.59 ms | 83.05% | 83.66% | 83.01% |
| **EfficientNet-B0 (Augmented)** | Albumentations Domain | 4,015,234 | 15.32 MB | 16.15 ms | 80.23% | 78.90% | 80.12% |
| **SparkNet-A (Ablation)** | None (Deterministic) | ~525,000 | 3.83 MB | 14.80 ms | 79.66% | 78.20% | 79.50% |

> [!WARNING]
> **Augmentation Finding**: Incorporating domain-specific geometric and color augmentations during training **reduced** overall test accuracy from 85.31% to 80.23% and Macro F1 from 84.93% to 78.90%. Due to the specialized visual characteristics of PV defects (where color nuances and precise orientations distinguish clean glass reflections from uniform dust or burn marks), excessive synthetic transformation degraded the network's discriminative boundaries on this dataset. Consequently, the non-augmented EfficientNet-B0 baseline was retained as the production checkpoint.

---

## 18. Robustness Testing

The production EfficientNet-B0 model was subjected to systematic perturbation stress-testing across 10 environmental and sensor degradation conditions (177 test samples per condition):

| Perturbation Condition | Evaluated Accuracy | Macro F1 | Relative Impact | Key Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Original Benchmark** | 83.62% | 82.35% | Baseline | Unmodified test set baseline |
| **Reduced Resolution** | **61.02%** | **58.59%** | **-22.60%** | **Strongest degradation condition**; fine crack lines and micro-burns vanish |
| **Gaussian Blur** | **76.84%** | **75.32%** | **-6.78%** | Noticeable degradation; softens sharp defect edges |
| **Low Light** | **78.53%** | **78.58%** | **-5.09%** | Reduced contrast challenges dark bird-drop and shadow distinction |
| **Low Contrast** | 81.36% | 81.39% | -2.26% | Slight drop; model relies on texture gradients |
| **High Contrast** | 81.92% | 82.49% | -1.70% | Minimal impact; preserved boundary edges |
| **High Brightness** | 82.49% | 82.84% | -1.13% | Minor impact; robust against glare |
| **Small Rotation ($\pm 10^\circ$)** | 83.05% | 83.06% | -0.57% | Negligible change; orientation-tolerant |
| **JPEG Compression** | 84.18% | 84.52% | +0.56% | Stable; resistant to standard lossy web transmission |
| **Sensor Noise** | 84.75% | 84.84% | +1.13% | Stable; high-frequency Gaussian noise does not disrupt feature maps |

---

## 19. Project Results

### Production EfficientNet-B0 Per-Class Test Breakdown (177 Test Images)

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Bird-drop** | 77.8% | 85.4% | 81.4% | 41 |
| **Clean** | 85.4% | 89.7% | 87.5% | 39 |
| **Dusty** | 81.1% | 81.1% | 81.1% | 37 |
| **Electrical-damage** | 90.9% | 95.2% | 93.0% | 21 |
| **Physical-damage** | 88.9% | 61.5% | 72.7% | 13 |
| **Snow-Covered** | 100.0% | 88.5% | 93.9% | 26 |
| **Overall / Macro** | **87.34%** | **83.57%** | **84.93%** | **177** |

---

## 20. Installation

### Prerequisites
- Python 3.12+
- Node.js 20+ & npm 10+
- Git

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/solar-panel-ai.git
cd solar-panel-ai
```

### 2. Set Up Python Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies
```bash
npm --prefix frontend install
```

### 5. Configure Environment Variables
```powershell
Copy-Item .env.example .env
```

---

## 21. Running Locally

### Start Backend API
```powershell
# From project root with .venv activated
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger UI Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

### Start Frontend UI
```powershell
# In a separate terminal
npm --prefix frontend run dev
```
- Web Application: [http://localhost:3000](http://localhost:3000)

---

## 22. Running with Docker

Run the entire system in isolated production containers:

```powershell
# Build both images
docker compose build

# Start containers in background
docker compose up -d

# Verify container health status
docker compose ps

# View unified service logs
docker compose logs -f

# Teardown containers
docker compose down
```
For comprehensive container operation instructions, refer to [`DOCKER.md`](DOCKER.md).

---

## 23. Testing

The repository includes 169 automated regression tests validating unit modules, integration flows, and deployment configurations:

```powershell
# Run full test suite with verbose output
python -m unittest discover -s tests -p "test_*.py" -v

# Run frontend production build validation
npm --prefix frontend run build

# Run frontend ESLint check
npm --prefix frontend run lint

# Validate Docker Compose configurations
docker compose config
```

---

## 24. Project Structure

```
solar-panel-ai/
├── configs/
│   └── config.yaml                     # Central project and hyperparameter configuration
├── data/                               # Dataset directory scaffolding (images git-ignored)
│   ├── processed/                      # Normalized dataset splits
│   ├── raw/                            # Original uncurated images
│   ├── test/                           # 177 independent test images
│   ├── train/                          # 528 training images
│   └── val/                            # 180 validation images
├── frontend/                           # Next.js 16 Web Application
│   ├── src/
│   │   ├── app/                        # App router pages (dashboard, new-inspection, etc.)
│   │   ├── components/                 # Reusable UI components (AppShell, Disclaimers)
│   │   └── lib/                        # API client and mock data
│   ├── Dockerfile                      # Multi-stage production container blueprint
│   ├── package.json                    # Node dependencies
│   └── tsconfig.json                   # TypeScript configuration
├── models/
│   └── checkpoints/                    # Model weights and checksums
│       ├── checkpoint_manifest.json    # Cryptographic hashes and architecture metadata
│       ├── efficientnet_b0_baseline_best.pth   # Primary production weights (46.42 MB)
│       ├── mobilenetv2_baseline_best.pth       # MobileNetV2 weights (25.90 MB)
│       ├── efficientnet_b0_augmented_best.pth  # Augmented experiment weights (46.42 MB)
│       └── sparkneta_baseline_best.pth         # SparkNet-A weights (3.83 MB)
├── notebooks/                          # Exploratory data analysis notebooks
├── results/                            # Evaluation artifacts and empirical figures
│   ├── gradcam/                        # Sample visual attention heatmaps
│   ├── metrics/                        # JSON reports, history CSVs, and split manifests
│   ├── plots/                          # Confusion matrices and training curves
│   └── predictions/                    # Misclassification audits and error logs
├── scripts/                            # Operational automation scripts
│   ├── acquire_dataset.py              # Kaggle dataset downloader
│   ├── analyze_fault_region.py         # Morphological RoI extraction
│   ├── deploy.sh                       # Production EC2 deployment script
│   ├── ec2_bootstrap.sh                # Ubuntu VM bootstrapping script
│   ├── estimate_severity.py            # Rule-based severity evaluation
│   ├── evaluate_baseline.py            # MobileNetV2 test evaluator
│   ├── evaluate_efficientnet.py        # EfficientNet-B0 test evaluator
│   ├── evaluate_robustness.py          # Perturbation testing benchmark
│   ├── final_model_comparison.py       # Comparative benchmarking script
│   ├── generate_gradcam.py             # Saliency map generator
│   ├── run_system_tests.py             # Comprehensive test runner
│   ├── setup_database.py               # Database migration and seeding
│   ├── split_dataset.py                # Hash-deduplicated dataset splitter
│   └── train_efficientnet.py           # EfficientNet-B0 training loop
├── src/                                # Core Python source library
│   ├── api/                            # FastAPI routes, schemas, and services
│   ├── data/                           # Dataset loaders and EDA routines
│   ├── database/                       # SQLAlchemy models and session engine
│   ├── evaluation/                     # Metric calculation and plotting utilities
│   ├── explainability/                 # Grad-CAM implementation
│   ├── maintenance/                    # Maintenance recommendation engine
│   ├── metadata/                       # Panel asset tracking manager
│   ├── models/                         # PyTorch model definitions
│   ├── preprocessing/                  # Standardization and augmentation pipelines
│   ├── robustness/                     # Perturbation transform engine
│   ├── severity/                       # Multi-factor severity triage logic
│   ├── training/                       # Training routines and loss handlers
│   └── utils/                          # Configuration and path resolution helpers
├── tests/                              # 19 test modules (169 regression tests)
├── .dockerignore                       # Docker context exclusions
├── .env.example                        # Template environment variables
├── .gitignore                          # Git repository ignore rules
├── AWS.md                              # EC2 deployment architecture and instructions
├── DOCKER.md                           # Local Docker container manual
├── Dockerfile                          # Multi-stage backend container blueprint
├── docker-compose.yml                  # Local development compose file
├── docker-compose.ec2.yml              # EC2-optimized compose file
├── requirements.txt                    # Python package dependencies
└── README.md                           # Project technical documentation
```

---

## 25. Limitations

- **Image Resolution Dependency**: Downsampling below $112 \times 112$ px degrades accuracy significantly (-22.6%), as fine micro-cracks and minor burns require sufficient pixel density.
- **Visual Saliency vs. Segmentation**: Grad-CAM outputs coarse class-activation maps rather than pixel-accurate defect boundaries.
- **Surface Appearance vs. Electrical Status**: Visual models detect surface manifestations of faults. Internal bypass diode failures or micro-cracks without visible discoloration cannot be identified from RGB imagery alone.
- **Dataset Class Imbalance**: Classes such as `Physical-damage` (69 total images) have lower representation than `Bird-drop` (207 images), resulting in lower recall on physical fractures.
- **Visual Severity Proxy**: Severity is an operational proxy based on visual coverage and classification certainty; it does not measure true power loss or internal thermal dissipation.

---

## 26. Future Improvements

- **Multimodal Sensor Integration**: Pairing standard RGB imagery with calibrated thermal infrared (FLIR) imagery to detect sub-surface hot-spots and bypassed diode strings.
- **Instance Segmentation**: Training a Mask R-CNN or YOLOv11-seg model for pixel-level polygon segmentation of cracks and debris.
- **Automated UAV Path Planning**: Edge deployment of quantized ONNX models aboard drone microcontrollers (e.g., NVIDIA Jetson) for autonomous real-time aerial survey flights.
- **Active Learning Loop**: Incorporating technician inspection feedback in the database to retrain on hard false positives and edge cases.
- **Managed Cloud Scaling**: Migration from standalone EC2 instances to AWS ECS/Fargate with Amazon RDS PostgreSQL for horizontally autoscaling commercial arrays.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
