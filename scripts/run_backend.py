import sys
from pathlib import Path
import uvicorn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, log_level="info")
