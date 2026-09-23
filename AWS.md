# AWS EC2 Deployment Guide — Solar Panel AI Inspection System

This guide outlines the production deployment configuration for the containerized **Solar Panel AI Inspection System** (FastAPI backend + Next.js frontend + SQLite persistence) to an **AWS EC2 Ubuntu instance** using Docker and Docker Compose.

> [!IMPORTANT]
> **Deployment Status**: EC2 deployment configuration was prepared and locally validated. Live AWS deployment was not performed because AWS credentials/EC2 access were unavailable and the project intentionally avoids cloud costs at this stage.

---

## 1. Architecture Overview

```
Public Internet
    ├── Port 22    : SSH (restricted to admin IP)
    ├── Port 3000  : Next.js Frontend (public web traffic)
    └── Port 8000  : FastAPI Backend & ML Engine (browser direct API calls)

AWS EC2 Instance (Ubuntu 24.04 LTS, t3.medium)
    ├── solar-panel-ai-frontend (:3000)
    │     └── Baked-in NEXT_PUBLIC_API_BASE_URL=http://<EC2_PUBLIC_IP>:8000
    └── solar-panel-ai-backend (:8000)
          └── Host Volume: /home/ubuntu/solar-panel-ai/data -> /app/data (SQLite)
```

---

## 2. Recommended AWS EC2 Configuration

| Parameter | Recommended Value | Rationale |
|:---|:---|:---|
| **Instance Type** | `t3.medium` | 2 vCPUs, 4.0 GiB RAM. PyTorch CPU inference on EfficientNet-B0 requires ~1.2–1.5 GiB peak memory during startup/warmup and batch processing. 4 GiB provides solid headroom for both Node.js and FastAPI without swapping. |
| **Operating System** | Ubuntu Server 24.04 LTS (HVM), SSD Volume Type | Standard, stable Linux distribution with immediate Docker CE support. |
| **Architecture** | 64-bit (x86_64) | Matches standard PyTorch CPU and pre-built wheels. |
| **Storage (EBS)** | 30 GiB gp3 (3000 IOPS, 125 MB/s) | Accommodates OS (~3 GB), Docker images (~4 GB total for backend + frontend build cache), test datasets, and database volume. |
| **Elastic IP** | Recommended | Allocates a persistent public IPv4 address so browser URLs remain constant across EC2 stops/starts. |

---

## 3. AWS Security Group Rules

Create a dedicated Security Group (e.g., `solar-panel-ai-sg`) with the following Inbound Rules:

| Type | Protocol | Port Range | Source | Purpose |
|:---|:---|:---|:---|:---|
| **SSH** | TCP | `22` | `My IP` (`<your-ip>/32`) | Secure administration (do not expose to 0.0.0.0/0). |
| **Custom TCP** | TCP | `3000` | `Anywhere-IPv4` (`0.0.0.0/0`) | Access to Next.js Web Frontend. |
| **Custom TCP** | TCP | `8000` | `Anywhere-IPv4` (`0.0.0.0/0`) | Access to FastAPI Backend & Swagger Docs (`/docs`). Required because client browser queries the API directly. |

> [!IMPORTANT]
> Port 8000 must be opened to public web traffic because client-side browser code (e.g. file upload and inspection trigger) issues HTTP requests directly from the client machine to the backend URL (`http://<EC2_PUBLIC_IP>:8000`).

---

## 4. EC2 Provisioning Steps

### Option A: Using User Data (Automated Bootstrap)
When launching the instance in the AWS Management Console:
1. Under **Advanced details**, expand **User data**.
2. Paste the contents of [`scripts/ec2_bootstrap.sh`](file:///scripts/ec2_bootstrap.sh).
3. Launch instance. Docker Engine and Compose will be pre-installed upon boot.

### Option B: Manual Provisioning via SSH
1. Connect to the EC2 instance from PowerShell:
   ```powershell
   ssh -i "path\to\your-key.pem" ubuntu@<EC2_PUBLIC_IP>
   ```
2. Run the bootstrap script or install Docker manually:
   ```bash
   sudo apt-get update -y
   sudo apt-get install -y curl git ca-certificates gnupg
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update -y
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

---

## 5. Transferring Project Files to EC2

From your local Windows machine (PowerShell):

```powershell
# Archive or rsync project files (excluding .venv, node_modules, and cache dirs)
# Example using tar + scp:
tar --exclude='.venv' --exclude='node_modules' --exclude='.next' --exclude='__pycache__' -czf solar-panel-ai.tar.gz .
scp -i "path\to\your-key.pem" solar-panel-ai.tar.gz ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/

# SSH into EC2 and extract:
ssh -i "path\to\your-key.pem" ubuntu@<EC2_PUBLIC_IP>
mkdir -p /home/ubuntu/solar-panel-ai
tar -xzf /home/ubuntu/solar-panel-ai.tar.gz -C /home/ubuntu/solar-panel-ai
cd /home/ubuntu/solar-panel-ai
```

---

## 6. Environment Configuration & Deployment

On the EC2 instance:

```bash
cd /home/ubuntu/solar-panel-ai

# Make scripts executable
chmod +x scripts/*.sh

# Run the automated deployment script with your EC2 Public IP:
bash scripts/deploy.sh <EC2_PUBLIC_IP>
```

The script will automatically:
1. Configure `.env` with `NEXT_PUBLIC_API_BASE_URL=http://<EC2_PUBLIC_IP>:8000`.
2. Build the Docker images passing the public URL as build args (ensuring Next.js inlines the correct URL).
3. Start the containers using `docker compose -f docker-compose.yml -f docker-compose.ec2.yml up -d`.
4. Wait for the backend health check to pass.

---

## 7. Verification & Testing

### 7.1 Container Status
```bash
docker compose ps
```
Both `solar-panel-ai-backend` and `solar-panel-ai-frontend` should show `Up (healthy)`.

### 7.2 Backend Health Check
```bash
curl http://localhost:8000/api/health
# Expected: {"status":"ok","service":"solar-panel-ai-api"}
```

### 7.3 Frontend HTTP Response
```bash
curl -I http://localhost:3000
# Expected: HTTP/1.1 200 OK
```

### 7.4 Live ML Inspection Test
Run a test inference with an existing sample image:
```bash
curl -X POST http://localhost:8000/api/inspect \
  -F "panel_id=EC2-TEST-PANEL-01" \
  -F "location=AWS Array 1" \
  -F "file=@data/test/Bird-drop/13.JPG"
```
Verify the JSON output includes:
- `predicted_class`: `"Bird-drop"`
- `confidence`: `~0.92`
- `gradcam_heatmap_path`: Valid path/base64
- `severity`: `"HIGH"`
- `recommendation`: Non-empty actionable maintenance advice

### 7.5 SQLite Persistence Verification
```bash
# Check initial inspection record count
curl -s http://localhost:8000/api/inspections | grep -o '"id":' | wc -l

# Restart backend container
docker compose restart backend

# Verify record persists
curl -s http://localhost:8000/api/inspections
```

---

## 8. Web Browser Validation

Once deployed, open your local browser:

1. **Dashboard**: Navigate to `http://<EC2_PUBLIC_IP>:3000`
2. **New Inspection**: Go to `http://<EC2_PUBLIC_IP>:3000/new-inspection`
3. **Upload**: Select any solar panel image from `data/test/` and trigger inspection.
4. **Results**: Review the Grad-CAM visualization, severity triage, and recommendations at `http://<EC2_PUBLIC_IP>:3000/inspection-result/<ID>`.
5. **History**: Verify recorded history at `http://<EC2_PUBLIC_IP>:3000/inspection-history`.

---

## 9. Troubleshooting

| Issue | Root Cause | Solution |
|:---|:---|:---|
| Frontend cannot fetch API (`NetworkError` / Failed to fetch) | Browser cannot reach port 8000 or wrong IP baked into frontend. | 1. Check AWS Security Group has port 8000 open.<br>2. Verify `NEXT_PUBLIC_API_BASE_URL` in `.env` is set to `http://<EC2_PUBLIC_IP>:8000` and rebuild frontend with `bash scripts/deploy.sh <EC2_PUBLIC_IP>`. |
| Backend unhealthy or crash loop | OOM (Out of Memory) during PyTorch initialization. | Ensure instance is `t3.medium` (4 GiB RAM) or higher. Check memory usage with `free -h`. |
| Changes in SQLite database lost after container restart | Data volume not mounted properly. | Check `docker-compose.yml` has `./data:/app/data` volume mounted and `/home/ubuntu/solar-panel-ai/data` exists. |
| Permission denied on Docker commands | User not in `docker` group. | Run `sudo usermod -aG docker ubuntu && newgrp docker`. |

---

## 10. Instance Shutdown & Cost Control

To avoid ongoing charges when testing is finished:

```bash
# Stop containers
cd /home/ubuntu/solar-panel-ai
docker compose down

# Stop the EC2 instance from AWS Management Console or AWS CLI:
# aws ec2 stop-instances --instance-ids <INSTANCE_ID>
```
To permanently delete resources, terminate the instance and release the associated Elastic IP.
