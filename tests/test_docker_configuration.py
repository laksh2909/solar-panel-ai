"""
Tests for Phase 19: Docker Containerization Configuration.

Validates Dockerfiles, docker-compose.yml, .dockerignore, environment configurations,
health check endpoints, and checkpoint protection without requiring Docker daemon execution.
"""

from pathlib import Path
import unittest
import yaml


class TestDockerConfiguration(unittest.TestCase):
    """Test suite validating Docker and Compose configuration artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.backend_dockerfile = cls.root / "Dockerfile"
        cls.frontend_dockerfile = cls.root / "frontend" / "Dockerfile"
        cls.compose_file = cls.root / "docker-compose.yml"
        cls.root_dockerignore = cls.root / ".dockerignore"
        cls.frontend_dockerignore = cls.root / "frontend" / ".dockerignore"
        cls.env_example = cls.root / ".env.example"
        cls.api_file = cls.root / "frontend" / "src" / "lib" / "api.ts"

    def test_dockerfiles_exist(self):
        """Verify that both backend and frontend Dockerfiles exist."""
        self.assertTrue(
            self.backend_dockerfile.exists(),
            f"Backend Dockerfile missing at {self.backend_dockerfile}",
        )
        self.assertTrue(
            self.frontend_dockerfile.exists(),
            f"Frontend Dockerfile missing at {self.frontend_dockerfile}",
        )

    def test_docker_compose_exists_and_valid_yaml(self):
        """Verify docker-compose.yml exists and has valid YAML syntax."""
        self.assertTrue(self.compose_file.exists(), "docker-compose.yml does not exist")
        with open(self.compose_file, "r", encoding="utf-8") as f:
            compose_data = yaml.safe_load(f)
        self.assertIsInstance(compose_data, dict)
        self.assertIn("services", compose_data)
        self.assertIn("backend", compose_data["services"])
        self.assertIn("frontend", compose_data["services"])

    def test_ports_configured(self):
        """Verify standard service ports 8000 and 3000 are mapped in docker-compose.yml."""
        with open(self.compose_file, "r", encoding="utf-8") as f:
            compose_data = yaml.safe_load(f)

        backend_ports = compose_data["services"]["backend"].get("ports", [])
        frontend_ports = compose_data["services"]["frontend"].get("ports", [])

        self.assertTrue(
            any("8000" in str(p) for p in backend_ports),
            "Backend port 8000 not properly exposed in docker-compose.yml",
        )
        self.assertTrue(
            any("3000" in str(p) for p in frontend_ports),
            "Frontend port 3000 not properly exposed in docker-compose.yml",
        )

    def test_healthchecks_configured(self):
        """Verify backend healthcheck targets /api/health and frontend depends on it."""
        with open(self.compose_file, "r", encoding="utf-8") as f:
            compose_data = yaml.safe_load(f)

        backend_hc = compose_data["services"]["backend"].get("healthcheck", {})
        self.assertIn("test", backend_hc)
        self.assertTrue(
            any("/api/health" in str(part) for part in backend_hc["test"]),
            "Backend healthcheck does not target /api/health",
        )

        frontend_deps = compose_data["services"]["frontend"].get("depends_on", {})
        self.assertIn("backend", frontend_deps)
        self.assertEqual(frontend_deps["backend"].get("condition"), "service_healthy")

    def test_data_persistence_volume_configured(self):
        """Verify SQLite data persistence volume is mounted for the backend."""
        with open(self.compose_file, "r", encoding="utf-8") as f:
            compose_data = yaml.safe_load(f)

        volumes = compose_data["services"]["backend"].get("volumes", [])
        self.assertTrue(
            any("/app/data" in str(v) for v in volumes),
            "Backend does not mount /app/data volume for persistence",
        )

    def test_dockerignore_protects_checkpoints(self):
        """Verify .dockerignore exists and explicitly preserves models/checkpoints."""
        self.assertTrue(self.root_dockerignore.exists(), ".dockerignore missing")
        content = self.root_dockerignore.read_text(encoding="utf-8")

        # Must ignore virtual environments and cache
        self.assertIn(".venv", content)
        self.assertIn("__pycache__", content)

        # Must explicitly whitelist checkpoints
        self.assertIn("!models/checkpoints", content)

    def test_backend_dockerfile_contents(self):
        """Verify backend Dockerfile uses Python 3.12, installs uvicorn, and sets CMD."""
        content = self.backend_dockerfile.read_text(encoding="utf-8")
        self.assertIn("python:3.12", content)
        self.assertIn("uvicorn", content)
        self.assertIn("EXPOSE 8000", content)
        self.assertIn("/api/health", content)

    def test_frontend_dockerfile_contents(self):
        """Verify frontend Dockerfile uses multi-stage Node 20, builds, and exposes 3000."""
        content = self.frontend_dockerfile.read_text(encoding="utf-8")
        self.assertIn("node:20-alpine", content)
        self.assertIn("npm run build", content)
        self.assertIn("EXPOSE 3000", content)
        # Should not use npm run dev in production container
        self.assertNotIn("npm run dev", content)

    def test_frontend_api_environment_aware(self):
        """Verify frontend api.ts supports NEXT_PUBLIC_API_BASE_URL and NEXT_PUBLIC_API_URL."""
        self.assertTrue(self.api_file.exists(), "frontend/src/lib/api.ts missing")
        content = self.api_file.read_text(encoding="utf-8")
        self.assertIn("NEXT_PUBLIC_API_BASE_URL", content)
        self.assertIn("NEXT_PUBLIC_API_URL", content)

    def test_env_example_documented(self):
        """Verify .env.example defines DATABASE_URL, NEXT_PUBLIC_API_BASE_URL and no secrets."""
        self.assertTrue(self.env_example.exists(), ".env.example missing")
        content = self.env_example.read_text(encoding="utf-8")
        self.assertIn("DATABASE_URL", content)
        self.assertIn("NEXT_PUBLIC_API_BASE_URL", content)
        # Check that no actual secret keys or credentials are leaked
        self.assertNotIn("AKIA", content)
        self.assertNotIn("BEGIN RSA PRIVATE KEY", content)


if __name__ == "__main__":
    unittest.main()
