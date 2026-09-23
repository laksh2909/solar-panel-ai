# Docker Deployment Guide — Solar Panel AI Inspection System

This guide outlines how to build, run, test, and manage the containerized **Solar Panel AI Inspection System** locally using Docker and Docker Compose on Windows PowerShell.

---

## 1. Architecture Overview

The system runs as two isolated, production-grade microservices communicating over a shared bridge network (`solar-net`):

```
                     +---------------------------------------+
                     |            Host Browser               |
                     |   (http://localhost:3000 -> UI)       |
                     |   (http://localhost:8000 -> REST API) |
                     +---------------------------------------+
                                  |                 |
                         Port 3000|         Port 8000
                                  v                 v
+-----------------------------------+     +-----------------------------------+
|      solar-panel-ai-frontend      |     |      solar-panel-ai-backend       |
| --------------------------------- |     | --------------------------------- |
| - Node 20 Alpine (Multi-stage)    |     | - Python 3.12 Slim                |
| - Next.js 16 Production Server    |     | - PyTorch CPU & OpenCV            |
| - SSR & Static Asset Serving      |     | - EfficientNet-B0 + Grad-CAM      |
| - Healthcheck: wget spider :3000  |     | - FastAPI REST Endpoints          |
|                                   |     | - Healthcheck: GET /api/health    |
+-----------------------------------+     +-----------------------------------+
                                                            |
                                                   Volume: ./data:/app/data
                                                            v
                                                  [solar_panel_ai.db (SQLite)]
```

---

## 2. Prerequisites

- **Docker Desktop** (version 24+ recommended, with WSL 2 backend).
- **Docker Compose** (v2+).
- **Windows PowerShell** or Windows Terminal.

Verify Docker is running:
```powershell
docker --version
docker compose version
```

---

## 3. Environment Configuration

Copy the example environment configuration:
```powershell
Copy-Item .env.example .env
```

Key environment variables:
- `NEXT_PUBLIC_API_BASE_URL`: Browser-accessible base URL for backend API requests (default: `http://localhost:8000`).
- `DATABASE_URL`: Database connection string. Defaults to SQLite at `sqlite:////app/data/solar_panel_ai.db` inside the container.
- For production PostgreSQL, set:
  `DATABASE_URL=postgresql+psycopg://username:password@db-host:5432/solar_panel_ai`

---

## 4. Build and Startup Commands

### Build Containers
Build both backend and frontend images:
```powershell
docker compose build
```

### Start Containers in Background
```powershell
docker compose up -d
```

### Check Container Status & Health
```powershell
docker compose ps
```
Both containers will report `(healthy)` when ready:
- `solar-panel-ai-backend`: listens on `http://localhost:8000`
- `solar-panel-ai-frontend`: listens on `http://localhost:3000`

---

## 5. Endpoints & UI Links

| Service | URL | Description |
| :--- | :--- | :--- |
| **Frontend UI** | [http://localhost:3000](http://localhost:3000) | Next.js Dashboard, New Inspection, History |
| **Backend Health** | [http://localhost:8000/api/health](http://localhost:8000/api/health) | System health and connection probe |
| **API Documentation** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive Swagger UI API Docs |
| **API Redoc** | [http://localhost:8000/redoc](http://localhost:8000/redoc) | Alternative OpenAPI documentation |

---

## 6. Real ML Inference Test via CLI

To verify that EfficientNet-B0 inference, Grad-CAM, fault-region analysis, and persistence execute inside the container, upload a test image:

```powershell
curl.exe -X POST "http://localhost:8000/api/inspect" `
  -F "panel_id=SP-DOCKER-TEST" `
  -F "location=Rooftop Array 1" `
  -F "file=@data/test/Bird-drop/13.JPG"
```

Expected JSON response:
```json
{
  "inspection_id": 23,
  "panel_id": "SP-DOCKER-TEST",
  "location": "Rooftop Array 1",
  "image_filename": "13.JPG",
  "predicted_class": "Bird-drop",
  "confidence": 0.9263,
  "visual_region_area_percent": 36.73,
  "severity": "HIGH",
  "urgency": "PRIORITY",
  "maintenance_action": "prioritize cleaning and inspection",
  "manual_inspection_recommended": true,
  "confidence_warning": null
}
```

---

## 7. Viewing Logs & Troubleshooting

View combined logs:
```powershell
docker compose logs -f
```

View backend logs only:
```powershell
docker compose logs -f backend
```

View frontend logs only:
```powershell
docker compose logs -f frontend
```

---

## 8. Data Persistence

The backend mounts the host directory `./data` to `/app/data` inside the container. This ensures that:
- `data/solar_panel_ai.db` persists across container restarts, stops, and rebuilds.
- Newly registered panels and inspection history records are retained.

To test persistence across restart:
```powershell
docker compose restart backend
curl.exe http://localhost:8000/api/panels/SP-DOCKER-TEST
```

---

## 9. Shutdown & Cleanup Commands

Stop containers:
```powershell
docker compose stop
```

Stop and remove containers, networks:
```powershell
docker compose down
```

Remove containers and build caches (if doing full fresh build):
```powershell
docker compose down --volumes --rmi local
```

---

## 10. Known Limitations & Notes

- **Port Conflicts**: Port `8000` must be available on the host machine. If another process is using port `8000`, stop it or reassign `ports` in `docker-compose.yml`.
- **Client-Side Networking**: Next.js client-side requests originate from the host browser, so `NEXT_PUBLIC_API_BASE_URL` must point to a host-reachable address (`http://localhost:8000` or public IP/domain), rather than container DNS names like `http://backend:8000`.
- **Database Engine**: Local containerization defaults to persistent SQLite. For cloud environments (Phase 20+), set `DATABASE_URL` to a managed PostgreSQL instance.
