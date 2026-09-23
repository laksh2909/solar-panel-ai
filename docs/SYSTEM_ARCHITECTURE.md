# System Architecture

## 1. High-Level Architectural Design

The **Solar Panel AI Inspection System** is structured as a decoupled, multi-tier microservices platform. The system separates user presentation, application business logic, deep learning inference, explainability routines, and relational persistence into modular layers.

```mermaid
graph TD
    Client["User / Client Web Browser"] -->|"HTTP / Web Interface (:3000)"| Frontend["Next.js 16 Frontend Web Application"]
    Client -->|"Direct REST API Requests (:8000)"| Backend["FastAPI REST Backend Service"]
    Frontend -.->|"SSR / Client Fetch"| Backend

    subgraph Backend_Application["FastAPI Core Service (:8000)"]
        Router["API Routing & Pydantic Validation"]
        Service["Inspection Service Orchestrator"]
        Router --> Service

        subgraph ML_CV_Subsystem["Machine Learning & Vision Engine"]
            Prep["Preprocessing & Validation (OpenCV / Albumentations)"]
            Model["PyTorch Deep Learning Engine (EfficientNet-B0)"]
            GradCAM["Grad-CAM Saliency Generator (features.8)"]
            RoI["Morphological Fault-Region Extractor"]
            Severity["Rule-Based Severity Estimator"]
            Maint["Maintenance Recommendation Engine"]

            Service --> Prep
            Prep --> Model
            Model --> GradCAM
            GradCAM --> RoI
            RoI --> Severity
            Severity --> Maint
        end

        subgraph Storage_Subsystem["Data Persistence Layer"]
            Repo["SQLAlchemy Repository Engine"]
            SQLite[("Local SQLite Database (solar_panel_ai.db)")]
            Postgres[("Enterprise PostgreSQL (Optional Cloud DB)")]
            
            Maint --> Repo
            Repo --> SQLite
            Repo -.->|DATABASE_URL| Postgres
        end
    end

    Repo -->|"Inspection History & Audit Trail"| Frontend
```

---

## 2. Component Details

### 2.1 Presentation Tier: Next.js 16 Web Application
- **Framework**: Next.js 16.3.5 (React 19, TypeScript, TailwindCSS v4).
- **Architecture**: Modular App Router design with static and dynamic routes:
  - `/dashboard`: Operational overview, fault class distribution bar charts, severity breakdown, and system status widgets.
  - `/new-inspection`: Interactive inspection studio with drag-and-drop image upload, panel ID assignment, and location metadata capture.
  - `/inspection-result/[id]`: Comprehensive diagnostic viewer featuring side-by-side original image and Grad-CAM heatmap overlay, visual region boundary metrics, severity badges, and maintenance cards.
  - `/inspection-history`: Searchable, filterable audit ledger with class, severity, and date filtering.
  - `/panel-details/[panelId]`: Dedicated asset tracking page displaying physical panel metadata and complete lifecycle inspection logs.
- **Client-Side Networking**: Direct asynchronous browser communication with backend REST endpoints via `NEXT_PUBLIC_API_BASE_URL`.

### 2.2 Application Tier: FastAPI REST Backend
- **Framework**: FastAPI (Python 3.12, Uvicorn ASGI server).
- **Core Endpoints**:
  - `POST /api/inspect`: Accepts multipart image upload, panel identifier, and location metadata; runs inference, Grad-CAM, region extraction, severity estimation, and persistence; returns comprehensive inspection JSON.
  - `GET /api/inspections`: Retrieves paginated inspection history with optional class and severity filters.
  - `GET /api/inspections/{id}`: Retrieves complete inspection report by ID.
  - `GET /api/panels`: Lists all registered physical solar panel assets.
  - `GET /api/panels/{panel_id}`: Retrieves asset details and historical inspection records for a specific panel.
  - `GET /api/health`: Service health probe reporting model initialization status, device hardware (CPU), and database connectivity.
- **Data Validation**: Strict Pydantic models enforcing schema validation and type safety.
- **API Documentation**: Automated OpenAPI/Swagger documentation exposed at `/docs` and ReDoc at `/redoc`.

### 2.3 Machine Learning & Computer Vision Tier
- **Deep Learning Framework**: PyTorch (CPU-optimized build) and torchvision.
- **Production Classifier**: EfficientNet-B0 transfer learning model (ImageNet-1K pretrained base, custom 6-class linear classification head, 4,015,234 parameters).
- **Explainability**: Custom Grad-CAM module hooking activations and gradients of terminal convolutional block (`features.8`).
- **Computer Vision Operations**: OpenCV (`cv2`) for bilateral resizing, color space standardization, binary thresholding, morphological closing/opening, and contour analysis.

### 2.4 Data Persistence Tier
- **ORM Layer**: SQLAlchemy declarative mapping separating domain entities from database drivers.
- **Default Local Storage**: SQLite file-based database (`data/solar_panel_ai.db`).
- **Production PostgreSQL Compatibility**: Architecture uses dialect-agnostic queries and psycopg driver compatibility, allowing seamless migration to PostgreSQL by setting the `DATABASE_URL` environment variable:
  ```bash
  DATABASE_URL=postgresql+psycopg://username:password@db-host:5432/solar_panel_ai
  ```

---

## 3. Containerization & Deployment Blueprints

### 3.1 Docker & Docker Compose Architecture
The system is fully containerized into two independent, lightweight microservices connected via an isolated bridge network (`solar-net`):

```
+-----------------------------------------------------------------------------------+
| Host System (Windows / Linux / macOS)                                             |
|                                                                                   |
|   Port 3000 ──► solar-panel-ai-frontend                                           |
|                 • Node 20 Alpine (Multi-stage build)                              |
|                 • Standalone Next.js production server                            |
|                 • Built-in healthcheck: wget spider :3000                         |
|                                                                                   |
|   Port 8000 ──► solar-panel-ai-backend                                            |
|                 • Python 3.12 Slim (Multi-stage build)                            |
|                 • PyTorch CPU & OpenCV headless libraries                         |
|                 • FastAPI / Uvicorn ASGI server                                   |
|                 • Built-in healthcheck: curl -f :8000/api/health                  |
|                 • Host Bind Mount: ./data ──► /app/data (Persistent SQLite DB)   |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### 3.2 AWS EC2 Deployment Blueprints
Deployment blueprints for Amazon Web Services (AWS) are prepared and documented in `AWS.md`:
- **Target Environment**: AWS EC2 virtual machine running Ubuntu Server 24.04 LTS (recommended: `t3.medium`, 2 vCPUs, 4 GiB RAM, 30 GiB gp3 EBS storage).
- **Automated Provisioning**: Shell scripts (`scripts/ec2_bootstrap.sh` and `scripts/deploy.sh`) to automate Docker installation, environment variable injection, and container startup.
- **Security Configuration**: Inbound security group rules documented for SSH (port 22, restricted to admin IP), Next.js frontend (port 3000, public), and FastAPI backend (port 8000, public).

> [!IMPORTANT]
> **Cloud Status**: The AWS EC2 deployment configuration was prepared and locally validated. Live AWS cloud deployment was **not** performed to avoid cloud expenses and because active AWS credentials were not provided.
