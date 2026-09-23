# Viva Notes — AI-Based Solar Panel Fault Detection and Inspection System

Concise questions and answers prepared for B.Tech AI & ML viva examination. Answers are intentionally brief, honest, and technically grounded in the actual project implementation.

---

## Q1. Why did you choose this project?

Solar energy is growing rapidly, but fault detection in large-scale solar installations is still done manually by technicians walking through fields or using aerial surveys. Manual inspection is slow, expensive, and inconsistent. AI-based image classification can automate fault identification, reduce response time, and scale to thousands of panels. This project directly addresses a real operational problem in renewable energy infrastructure.

---

## Q2. What problem does this project solve?

It automates the visual inspection of photovoltaic solar panel modules. Given a photograph of a solar panel, the system:
1. Classifies the panel's condition (clean, dusty, bird-drop, electrical damage, physical damage, or snow-covered).
2. Highlights which part of the image influenced the decision.
3. Estimates a visual severity tier.
4. Suggests maintenance action.
5. Logs the inspection into a persistent database tied to a panel asset.

---

## Q3. What are the six fault classes?

1. **Bird-drop** — Localized organic deposits from birds, causing hotspots.
2. **Clean** — No defects; normal operating state.
3. **Dusty** — Uniform or streaked particulate dust accumulation.
4. **Electrical-damage** — Cell burnout, burn marks, solder melting, bypass diode failure.
5. **Physical-damage** — Glass cracking, delamination, impact fractures.
6. **Snow-Covered** — Opaque snow or ice covering the panel surface.

---

## Q4. Why did you choose EfficientNet-B0?

EfficientNet-B0 uses compound scaling — it simultaneously scales network depth, width, and input resolution using fixed coefficients derived from a neural architecture search. This gives it stronger feature extraction capacity compared to older models of similar size. It achieved the best overall test accuracy (85.31%) and macro F1 (84.93%) among the models evaluated. It is also computationally feasible on CPU hardware at ~16 ms per image, which made testing and validation practical without a GPU cluster.

---

## Q5. Why did you compare with MobileNetV2?

MobileNetV2 is a lightweight, edge-optimized architecture that uses inverted residual blocks with linear bottlenecks. Including it in the comparison answers the question: "Can we achieve acceptable accuracy with fewer parameters and faster inference?" MobileNetV2 reached 83.05% accuracy and ran at ~11.59 ms per image — usable for resource-constrained applications where EfficientNet-B0's extra 2.26% accuracy gain does not justify the additional 1.78 million parameters.

---

## Q6. Why did data augmentation reduce performance instead of improving it?

This was the key finding from the augmentation ablation experiment. Adding domain-specific augmentations — including random rotations, brightness/contrast shifts, blur, and noise — during EfficientNet-B0 training lowered test accuracy from 85.31% to 80.23% and macro F1 from 84.93% to 78.90%.

The most likely reason is that in photovoltaic fault classification, many discriminative features are subtle and class-specific — for example, the dark brownish discolouration of an electrical burn, the spider-web line pattern of physical glass fracture, or the uniform matte texture of dust. Synthetic color shifts, angle rotations, and blur distortions can confound these subtle visual cues, causing the model to learn less precise decision boundaries. The augmentation strategy that works well for object recognition datasets (where shape is dominant) does not necessarily transfer to surface anomaly inspection tasks.

---

## Q7. What is Grad-CAM?

Grad-CAM (Gradient-weighted Class Activation Mapping) is an explainability technique for convolutional neural networks. It works by:
1. Running a forward pass to obtain the predicted class score.
2. Backpropagating the gradient of that score with respect to the feature maps of a target convolutional layer.
3. Global-average-pooling those gradients to produce neuron importance weights.
4. Computing a weighted sum of the feature maps and applying ReLU to suppress negative activations.
5. Upsampling the resulting 7×7 spatial map to the original image resolution.

The output is a heatmap highlighting which image regions drove the classification decision.

---

## Q8. Is Grad-CAM the same as image segmentation? Can it locate defects precisely?

No. Grad-CAM provides approximate visual saliency — it shows which broad image regions influenced the network's prediction. It is not pixel-level semantic segmentation. The output resolution originates from the network's final convolutional layer (7×7 feature map), which after upsampling produces a coarse spatial map, not sharp object boundaries. The system clearly labels Grad-CAM visualizations as visual attention maps, not certified defect boundaries.

---

## Q9. How is severity calculated in this system?

Severity is a rule-based heuristic combining three factors:
1. **Class inherent risk weight** — Electrical damage has the highest risk weight (1.00), physical damage (0.90), snow-covered (0.75), bird-drop (0.70), dusty (0.55), clean (0.00).
2. **Model confidence** — Higher confidence in a high-risk class increases severity.
3. **Approximate visual area** — The percentage of module surface covered by high Grad-CAM activation is normalized and incorporated.

These three factors produce a composite score mapped to five tiers: CRITICAL, HIGH, MEDIUM, LOW, or NEGLIGIBLE.

**Important**: This severity is a visual operational heuristic. It does not measure actual electrical power loss, temperature, or structural integrity.

---

## Q10. What role does prediction confidence play?

Confidence plays two roles:
1. **Input to severity scoring** — A high-confidence prediction of an electrical fault raises the severity composite score.
2. **Low-confidence override** — If the top-1 softmax probability falls below 0.60, the system flags a confidence warning and recommends manual inspection, regardless of the predicted class. This prevents high-stakes decisions from being based on uncertain model outputs.

---

## Q11. What is FastAPI and why did you use it?

FastAPI is a modern Python web framework built on Starlette and Pydantic. It uses Python type hints to generate automatic request/response validation, and automatically produces OpenAPI (Swagger) and ReDoc documentation. It supports asynchronous request handling via Python's `async/await`, which is important for concurrent image upload and inference tasks. It is also significantly faster than traditional synchronous frameworks like Flask. The FastAPI backend exposes all inspection, panel, and health endpoints, making it straightforward to extend or integrate.

---

## Q12. Why Next.js for the frontend?

Next.js provides a hybrid React framework with both server-side rendering (SSR) and static site generation (SSG). For this project, it enables:
- Fast initial page loads for the dashboard via SSR.
- A clean App Router structure organizing inspection, history, and panel detail pages.
- Built-in TypeScript support for type-safe API integration.
- Easy integration with TailwindCSS for responsive layout design.

---

## Q13. Why is the database architecture described as PostgreSQL-ready?

The database layer is implemented using SQLAlchemy, a Python ORM that abstracts over multiple database backends. All queries are written using dialect-agnostic SQLAlchemy constructs — no raw SQL strings are embedded. The database connection is entirely determined by the `DATABASE_URL` environment variable. Switching from SQLite to PostgreSQL requires only changing this single variable, with no code modifications. This makes the system immediately deployable to managed cloud databases like Amazon RDS or Google Cloud SQL.

---

## Q14. Why was SQLite used during development and testing?

SQLite requires zero server configuration — it is a self-contained file-based database. During development, this enables the entire system (backend, ML inference, and database) to run as a single Docker Compose stack without external dependencies. It is also the standard approach for validating application database logic before provisioning cloud database infrastructure. Live PostgreSQL configuration was not available during testing.

---

## Q15. What is Docker and why was it used?

Docker is a container platform that packages applications, their runtime dependencies, and their configuration into isolated, reproducible container images. For this project:
- The Python backend (including PyTorch, OpenCV, FastAPI, and all pip dependencies) is packaged into a `python:3.12-slim` container.
- The Next.js frontend is packaged into a multi-stage `node:20-alpine` container.
- Docker Compose orchestrates both containers on a shared bridge network with port mapping and persistent data volumes.

This ensures the system behaves identically regardless of whether it is run on a developer's Windows laptop, a Linux CI server, or a cloud virtual machine.

---

## Q16. Why Docker specifically for this project?

Three main reasons:
1. **Reproducibility**: PyTorch and OpenCV have complex native dependencies that are difficult to install consistently across operating systems. Docker eliminates "works on my machine" problems.
2. **Environment isolation**: The backend's CPU PyTorch build does not conflict with the system Python or other projects.
3. **Cloud portability**: The same Docker Compose configuration used locally is directly usable on any cloud VM by running `docker compose up -d`. AWS EC2 deployment configurations are already prepared in `docker-compose.ec2.yml`.

---

## Q17. Was the project deployed to AWS?

No. The AWS EC2 deployment configuration was **prepared and locally validated**. This includes:
- An automated EC2 Ubuntu 24.04 bootstrap script (`scripts/ec2_bootstrap.sh`) for installing Docker.
- An EC2-specific Docker Compose file (`docker-compose.ec2.yml`) with environment variable injection for the public EC2 IP.
- A deployment automation script (`scripts/deploy.sh`) that writes the `.env` file and starts containers.
- Full EC2 security group, instance sizing, and EBS storage recommendations documented in `AWS.md`.

Live AWS deployment was **not performed** because active AWS credentials and EC2 instance access were unavailable, and the project intentionally avoided cloud costs during this development phase.

---

## Q18. What are the limitations of this project?

1. **Dataset size is limited** — 885 total images.
2. **Class imbalance** — Physical damage has only 69 images, causing recall of 61.5%.
3. **CPU-only hardware** — Training and testing were conducted on CPU without GPU acceleration.
4. **Resolution sensitivity** — Accuracy drops sharply (-22.6%) with downsampled images.
5. **Grad-CAM is approximate** — Not a pixel-level segmentation of the defect.
6. **Severity is heuristic** — Does not measure actual power loss or temperature.
7. **Maintenance advice is workflow support** — Not a certified engineering diagnosis.
8. **No live user authentication** — The web application is a prototype without RBAC.
9. **Live AWS deployment not performed** — Cloud configuration is prepared but not live.
10. **RGB imagery only** — Cannot detect purely electrical or subsurface faults invisible to optical cameras.

---

## Q19. What was the biggest robustness failure mode?

**Reduced resolution**. When test images were downsampled to lower pixel density (simulating distant drone captures or thumbnail previews), model accuracy dropped from 83.62% to **61.02%** — a -22.60% degradation. This makes sense because physical crack lines, fine electrical burn edges, and the texture differences between clean glass and faint dust are all encoded in high-frequency detail that disappears under heavy downsampling. The practical implication is that inspection cameras must maintain sufficient resolution for reliable classification.

---

## Q20. What makes this different from the base SparkNet research paper?

The reference SparkNet paper (IEEE Access, 2025) focused on a custom dual-branch convolutional architecture (SparkNet-A) trained without data augmentation. This project extends that foundation in several dimensions:
- Benchmarks SparkNet-A against pre-trained ImageNet transfer learning architectures (EfficientNet-B0, MobileNetV2) under identical, leakage-free evaluation conditions.
- Adds Grad-CAM visual explainability to support field technician interpretation.
- Adds approximate visual fault-region bounding box and area extraction.
- Adds rule-based severity estimation and maintenance recommendation workflow.
- Adds panel asset metadata tracking and relational inspection persistence.
- Adds a complete web-based inspection interface (FastAPI + Next.js).
- Adds Docker containerization and AWS deployment blueprints.
- Adds systematic robustness testing under 10 environmental perturbation conditions.

---

## Q21. What is the final result of the best model?

The production **EfficientNet-B0** classifier evaluated on the 177 independent test images:
- **Test Accuracy: 85.31%**
- **Macro F1 Score: 84.93%**
- **Weighted F1 Score: 85.25%**
- **Macro Precision: 87.34%**
- **Macro Recall: 83.57%**
- **Model Size: 15.32 MB**
- **CPU Inference Latency: ~16.10 ms/image**

---

## Q22. What did you learn from the experiments?

1. **Augmentation is dataset-dependent** — What improves generalization in one domain can hurt performance in another where class distinctions rely on subtle surface textures rather than coarse geometric shapes.
2. **Evaluation pipeline standardization is critical** — A single implementation difference (OpenCV bilinear vs. PIL bicubic resizing) produced a 1.69% accuracy discrepancy on the same model and data.
3. **Explainability shapes operational trust** — Providing Grad-CAM overlays alongside predictions is essential for fault inspection systems; raw probability scores alone are insufficient for field personnel.
4. **System engineering matters as much as modeling** — Containerization, API design, database persistence, and frontend usability are what transform a research prototype into a deployable inspection tool.
5. **Robustness testing should be planned, not afterthought** — The reduced-resolution vulnerability was only discovered through systematic stress testing, yet it has major implications for real-world drone-captured imagery.
