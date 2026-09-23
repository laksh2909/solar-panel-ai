# Limitations and Future Scope

## 1. Project Limitations

To maintain academic and professional honesty, the limitations of the current implementation are explicitly documented across machine learning, computer vision, data, and deployment dimensions.

### 1.1 Machine Learning & Dataset Limitations
1. **Limited Dataset Size**: The benchmark dataset contains 885 total images across 6 classes. While sufficient for transfer learning evaluation, deep learning models benefit significantly from tens of thousands of diverse operational samples.
2. **Pronounced Class Imbalance**: The dataset distribution is non-uniform, ranging from 207 images for `Bird-drop` (23.4%) down to only 69 images for `Physical-damage` (7.8%). This limited the recall of physical fracture detection to 61.5% on the test set.
3. **Single Optical Sensor Modality**: The system operates exclusively on optical RGB photography. Visual RGB imagery cannot detect internal electrical faults that exhibit no surface browning or discoloration, such as deactivated bypass diodes, micro-cracks beneath cell busbars, or potential-induced degradation (PID).
4. **Resolution & Blur Sensitivity**: Controlled perturbation testing demonstrated that the model is vulnerable to downsampling below $112 \times 112$ px (-22.60% accuracy drop) and optical defocus blur (-6.78% accuracy drop). High-resolution, focused photography is essential for reliable inference.

### 1.2 Computer Vision & Explainability Limitations
1. **Approximate Saliency vs. Exact Defect Segmentation**: Grad-CAM outputs coarse $7 \times 7$ feature activation maps bilinearly interpolated to $224 \times 224$. It indicates the visual regions influencing the neural network's decision; it is **not** a pixel-accurate semantic defect segmentation.
2. **Approximate Fault-Region Boundaries**: The morphological bounding box and area coverage percentage ($A_{\text{percent}}$) are derived from thresholded Grad-CAM attention. They provide approximate region-of-interest localization for triage rather than certified engineering measurements of physical damage area.

### 1.3 Operational Heuristic Limitations
1. **Visual Severity Proxy**: Severity estimation is an AI-assisted visual heuristic combining class risk weights, model confidence, and visual area extent. It does **not** directly measure electrical power degradation (kW), internal p-n junction temperature ($^\circ\text{C}$), structural glass strain, or crack depth ($\mu\text{m}$).
2. **Workflow-Support Nature of Recommendations**: Maintenance actions are rule-based operational guidance intended to support O&M dispatch prioritization. They do not replace certified engineering safety protocols, manufacturer warranty inspections, or licensed electrician verification.

### 1.4 System & Deployment Limitations
1. **Development Hardware**: Training, benchmarking, and test suite execution were conducted entirely on standard CPU hardware. While inference latency is fast (~16.10 ms/image), high-throughput batch video processing would require GPU acceleration.
2. **Absence of User Authentication**: The current web application operates as an internal engineering prototype and does not implement role-based access control (RBAC), multi-tenant isolation, or OAuth2/JWT user authentication.
3. **Local Database Validation**: The persistence layer was validated using local SQLite storage. While architecturally PostgreSQL-ready via SQLAlchemy, live multi-node PostgreSQL replication was not evaluated during this phase.
4. **Cloud Deployment Status**: Amazon Web Services (AWS) EC2 deployment configurations (Ubuntu 24.04 bootstrap, systemd services, Docker Compose) were prepared and locally validated. Live AWS hosting was intentionally **not** executed to avoid cloud hosting charges and because active cloud credentials were not provided.

---

## 2. Future Scope

The following enhancements represent planned directions for future research and engineering development:

### 2.1 Advanced Vision & Multimodal Sensing (Future Work)
- **Multimodal Optical & Thermal (FLIR) Fusion**: Pairing RGB imagery with calibrated radiometric thermal infrared (IR) video. Thermal imagery immediately exposes bypassed cell strings, localized hot-spots, and PID without requiring visible surface discolouration.
- **Pixel-Level Semantic & Instance Segmentation**: Training dedicated segmentation networks (such as Mask R-CNN or YOLOv11-seg) on polygon-annotated datasets to delineate precise physical crack boundaries and calculate millimeter-accurate defect surface areas.
- **Electrical I-V Curve Telemetry Integration**: Fusing computer vision predictions with live supervisory control and data acquisition (SCADA) electrical sensor telemetry (string current, open-circuit voltage, inverter DC yield) for true power loss attribution.

### 2.2 Model & Dataset Enhancements (Future Work)
- **Expanded Multi-Site Field Dataset**: Expanding the training repository to include drone-captured imagery across diverse geographic climates, panel tilt angles, and varying ground albedo conditions (desert sand, agricultural grass, commercial rooftops).
- **Self-Supervised & Vision Transformer Pretraining**: Evaluating vision transformer (ViT) backbones and self-supervised masked autoencoders (MAE) pretrained on massive unlabeled aerial solar imagery.
- **Uncertainty Calibration**: Implementing temperature scaling or Monte Carlo Dropout to produce formal Bayesian uncertainty estimates for high-stakes electrical fault classifications.

### 2.3 Edge & Robotics Deployment (Future Work)
- **Edge Model Quantization & Compilation**: Quantizing the EfficientNet-B0 backbone to INT8 precision using ONNX Runtime or TensorRT for low-power microcontroller deployment aboard drone payloads (e.g., NVIDIA Jetson Orin Nano).
- **Real-Time UAV Flight Path Planning**: Integrating model inference directly with drone flight controllers to dynamically hover and re-photograph suspected anomaly regions at higher zoom.

### 2.4 Enterprise Software & Cloud Infrastructure (Future Work)
- **Enterprise Security**: Implementing OAuth2/OIDC user authentication, role-based permissions (Viewer, Field Technician, Plant Engineer), and encrypted HTTPS/TLS communication.
- **Managed Cloud Autoscaling**: Deploying containerized services on AWS ECS/Fargate with Amazon RDS PostgreSQL and AWS S3 image storage, backed by automated CloudWatch monitoring and alerting pipelines.
- **Active Learning & Feedback Loop**: Providing an interactive technician review portal in the web UI where field engineers can correct model predictions, automatically appending verified edge cases to a continuous retraining loop.
- **Native Mobile Application**: Building a cross-platform mobile application (React Native / Flutter) enabling off-grid field technicians to capture, inspect, and log panel anomalies on handheld devices.
