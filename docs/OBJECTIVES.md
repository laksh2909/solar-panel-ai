# Project Objectives

## 1. Primary Academic & Engineering Goals

The primary goal of this project is to develop, evaluate, and package an automated visual inspection and decision-support system for photovoltaic (PV) solar panels. The project bridges the gap between deep learning computer vision research and practical solar operations and maintenance (O&M) workflows.

---

## 2. Specific Technical Objectives

### Objective 1: Six-Class Photovoltaic Anomaly Classification
Design and implement an end-to-end computer vision pipeline to classify single-panel RGB images into six canonical operational conditions:
- `Bird-drop` (localized organic contamination)
- `Clean` (defect-free baseline module)
- `Dusty` (diffuse particulate deposition)
- `Electrical-damage` (cell burnout, interconnect degradation)
- `Physical-damage` (structural fractures, delamination)
- `Snow-Covered` (opaque ice/snow occlusion)

### Objective 2: Comparative Architecture Benchmarking
Implement and empirically compare transfer learning architectures representing distinct trade-offs between parameter efficiency and representation capacity:
- **MobileNetV2**: Lightweight baseline benchmark evaluated for resource-constrained edge feasibility.
- **EfficientNet-B0**: High-capacity compound-scaled convolutional backbone evaluated for maximum classification performance.
- **SparkNet-A**: Reference dual-branch convolutional architecture evaluated as an ablation study.

### Objective 3: Experimental Evaluation of Data Augmentation
Rigorously test the hypothesis that domain-specific geometric and color augmentations improve model generalization on photovoltaic defects. Measure test accuracy, macro F1, and per-class metrics to objectively determine whether synthetic transforms aid or disrupt feature discriminability.

### Objective 4: Robustness Evaluation Under Controlled Degradations
Systematically stress-test the production model against 10 controlled image degradation and sensor perturbation conditions (including reduced resolution, Gaussian blur, low-light illumination, high brightness, contrast variations, sensor noise, lossy compression, and small rotations) to identify operational failure modes.

### Objective 5: Visual Explainability via Grad-CAM
Incorporate Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the final convolutional layer of the network (`features.8`). Generate 2D visual attention heatmaps that highlight the spatial image regions that contributed most strongly to the model's prediction.

### Objective 6: Approximate Visual Fault-Region Extraction
Develop a computer vision contour extraction algorithm operating on the Grad-CAM saliency distribution to estimate:
- Spatial bounding boxes $(x, y, w, h)$ bounding the primary attention area.
- Normalized centroid coordinates $(\bar{x}, \bar{y})$ of the activated region.
- Approximate percentage of the total module surface area covered by high activation.

> [!NOTE]
> This extraction provides an approximate visual attention region for inspection triage; it is **not** a certified pixel-level defect segmentation.

### Objective 7: Rule-Based Visual Severity Triage
Construct a transparent, multi-factor heuristic scoring system that combines the predicted class risk weight, model confidence, and visual area extent into operational severity tiers: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, and `NEGLIGIBLE`.

> [!IMPORTANT]
> The severity grade is an AI-assisted visual heuristic proxy. It does not directly measure kilowatt electrical power loss, internal junction temperature, micro-crack depth, or structural load capacity.

### Objective 8: Prescriptive Maintenance Workflow Guidance
Implement a decision-logic engine mapping diagnostic outputs to standardized maintenance recommendations (e.g., immediate electrical dispatch, scheduled washing, seasonal monitoring, technician re-inspection) accompanied by urgency levels (`ROUTINE`, `SCHEDULED`, `PRIORITY`, `IMMEDIATE_REVIEW`).

### Objective 9: Asset Metadata Tracking
Associate individual inspection records with physical solar asset metadata, including unique Panel IDs, string/array identifiers, physical facility location strings, geographical coordinates (latitude/longitude), and commissioning dates.

### Objective 10: Relational Inspection History Persistence
Develop an enterprise-ready database schema using SQLAlchemy to persist panel records and inspection history logs. Validate the schema using a lightweight local SQLite database while maintaining full architectural compatibility with production PostgreSQL servers.

### Objective 11: Interactive Full-Stack Web Application
Engineer a modern, user-friendly microservices web interface:
- **FastAPI REST API**: High-throughput asynchronous backend service serving model inference, image processing, and database endpoints with interactive OpenAPI documentation.
- **Next.js 16 Web Application**: Responsive frontend providing operational dashboards, drag-and-drop inspection studios, side-by-side Grad-CAM visualization, and historical inspection search/filtering.

### Objective 12: Production Packaging and Deployment Preparation
Package the complete system into modular, multi-stage Docker containers using Docker Compose. Prepare and locally validate deployment configurations for cloud virtual machines (specifically Amazon Web Services EC2 Ubuntu instances) without incurring cloud hosting expenses.
