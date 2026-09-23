# References

This document lists the key references, frameworks, libraries, and datasets used in the project. Where exact bibliographic details (DOI, page numbers, publisher) could not be confirmed from project artifacts, they are marked with `[VERIFY]` for manual verification before submission.

---

## 1. Base Research Paper

**[1]** SparkNet Solar Panel Fault Detection  
Title: *SparkNet — A Solar Panel Fault Detection Deep Learning Model*  
Published in: IEEE Access, 2025  
`[VERIFY: Full author list, exact issue number, DOI, page range]`  
- This paper serves as the reference architecture for the **SparkNet-A** dual-branch convolutional model evaluated in this project.
- The paper also describes the benchmark solar panel image dataset and the six-class classification problem formulation.

---

## 2. Datasets

**[2]** Solar Panel Images Dataset  
Platform: Kaggle  
URL: [https://www.kaggle.com/datasets/pythonafroz/solar-panel-images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images)  
Contributor: pythonafroz  
Description: 885 labeled solar panel images across six condition classes (Bird-drop, Clean, Dusty, Electrical-damage, Physical-damage, Snow-Covered).  
`[VERIFY: Original data collection source, collection date, license terms]`

---

## 3. Deep Learning Frameworks & Model Architectures

**[3]** PyTorch  
Authors: Paszke, A., Gross, S., Massa, F., et al.  
Title: *PyTorch: An Imperative Style, High-Performance Deep Learning Library*  
Published in: Advances in Neural Information Processing Systems (NeurIPS), 2019  
URL: [https://pytorch.org](https://pytorch.org)  
`[VERIFY: Exact citation format, volume, page numbers]`

**[4]** torchvision  
Title: torchvision — PyTorch Image and Video Datasets, Transforms, and Model Zoo  
URL: [https://pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html)  
Used for: Pre-trained EfficientNet-B0, MobileNetV2 model weights (ImageNet-1K), and evaluation transforms.

**[5]** EfficientNet Compound Scaling  
Authors: Tan, M., Le, Q.V.  
Title: *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*  
Published in: International Conference on Machine Learning (ICML), 2019  
`[VERIFY: Proceedings volume, pages, arXiv preprint ID if citing preprint]`

**[6]** MobileNetV2  
Authors: Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C.  
Title: *MobileNetV2: Inverted Residuals and Linear Bottlenecks*  
Published in: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018  
`[VERIFY: Proceedings volume, pages, DOI]`

---

## 4. Explainability

**[7]** Grad-CAM: Gradient-weighted Class Activation Mapping  
Authors: Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.  
Title: *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*  
Published in: International Journal of Computer Vision (IJCV), 2020 (conference version: ICCV 2017)  
`[VERIFY: DOI, exact journal volume and issue number, page numbers]`  
URL: [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)

---

## 5. Image Processing & Augmentation

**[8]** OpenCV — Open Source Computer Vision Library  
Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.  
URL: [https://opencv.org](https://opencv.org)  
Version used: OpenCV 4.x  
Used for: Image loading, bilinear resizing, color space conversion, binary thresholding, morphological operations, contour extraction, and Grad-CAM heatmap generation.

**[9]** Albumentations  
Authors: Buslaev, A., Iglovikov, V.I., Khvedchenya, E., Parinov, A., Druzhinin, M., Kalinin, A.A.  
Title: *Albumentations: Fast and Flexible Image Augmentations*  
Published in: Information, 2020, Vol. 11, No. 2  
URL: [https://albumentations.ai](https://albumentations.ai)  
`[VERIFY: DOI, page numbers]`  
Used for: Domain-specific augmentation pipeline in `src/preprocessing/augmentation.py` and canonical deterministic resizing in evaluation.

---

## 6. Web Framework & API

**[10]** FastAPI  
Author: Sebastián Ramírez  
Title: FastAPI — Modern, Fast Web Framework for Building APIs with Python  
URL: [https://fastapi.tiangolo.com](https://fastapi.tiangolo.com)  
Used for: REST API backend, Pydantic validation, OpenAPI/Swagger documentation generation.

**[11]** Next.js  
Developer: Vercel  
Title: Next.js — The React Framework for the Web  
URL: [https://nextjs.org](https://nextjs.org)  
Version: 16.3.5  
Used for: Interactive web frontend with App Router (SSR and SSG), dashboard visualization, inspection studio.

---

## 7. Database & ORM

**[12]** SQLAlchemy  
Author: Michael Bayer  
Title: SQLAlchemy — The Database Toolkit for Python  
URL: [https://www.sqlalchemy.org](https://www.sqlalchemy.org)  
Used for: ORM schema definitions (`Panel`, `Inspection`), database engine abstraction, and SQLite/PostgreSQL compatibility.

---

## 8. Scientific Computing & Evaluation Libraries

**[13]** NumPy  
Authors: Harris, C.R., Millman, K.J., van der Walt, S.J., et al.  
Title: *Array programming with NumPy*  
Published in: Nature, 585, 357–362, 2020  
URL: [https://numpy.org](https://numpy.org)

**[14]** scikit-learn  
Authors: Pedregosa, F., Varoquaux, G., Gramfort, A., et al.  
Title: *Scikit-learn: Machine Learning in Python*  
Published in: Journal of Machine Learning Research (JMLR), 12, 2825–2830, 2011  
URL: [https://scikit-learn.org](https://scikit-learn.org)  
Used for: Classification metrics (`accuracy_score`, `f1_score`, `classification_report`, `confusion_matrix`).

**[15]** pandas  
Developers: The pandas Development Team  
Title: pandas — Powerful Python Data Analysis Library  
URL: [https://pandas.pydata.org](https://pandas.pydata.org)

---

## 9. Containerization & Infrastructure

**[16]** Docker  
Developer: Docker, Inc.  
Title: Docker — Platform for Developing, Shipping, and Running Applications  
URL: [https://www.docker.com](https://www.docker.com)

**[17]** Amazon Web Services EC2  
Developer: Amazon Web Services  
Title: Amazon Elastic Compute Cloud (EC2)  
URL: [https://aws.amazon.com/ec2](https://aws.amazon.com/ec2)  
Note: AWS EC2 deployment was prepared and locally validated. No live deployment was performed in this project.

---

## Notes on References
- References marked `[VERIFY]` require manual lookup to confirm exact bibliographic details (DOI, page numbers, authors) before formal submission.
- Do not use the approximate citation details above in formal academic bibliography submissions without cross-checking against the original published sources (IEEE Xplore, ACM Digital Library, arXiv, or Google Scholar).
