"""
Tests for Phase 20: AWS EC2 Deployment Configuration.

Validates deployment scripts, docker-compose overrides, environment documentation,
and security posture without requiring AWS credentials or a live EC2 instance.
"""

from pathlib import Path
import re
import unittest
import yaml


class TestAWSConfiguration(unittest.TestCase):
    """Test suite validating AWS EC2 deployment configuration artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.bootstrap_script = cls.root / "scripts" / "ec2_bootstrap.sh"
        cls.deploy_script = cls.root / "scripts" / "deploy.sh"
        cls.compose_ec2 = cls.root / "docker-compose.ec2.yml"
        cls.aws_md = cls.root / "AWS.md"
        cls.env_example = cls.root / ".env.example"
        cls.api_main = cls.root / "src" / "api" / "main.py"
        cls.compose_base = cls.root / "docker-compose.yml"
        cls.frontend_dockerfile = cls.root / "frontend" / "Dockerfile"

    # ------------------------------------------------------------------
    # Deployment script existence
    # ------------------------------------------------------------------

    def test_bootstrap_script_exists(self):
        """Verify ec2_bootstrap.sh exists."""
        self.assertTrue(
            self.bootstrap_script.exists(),
            f"ec2_bootstrap.sh missing at {self.bootstrap_script}",
        )

    def test_deploy_script_exists(self):
        """Verify deploy.sh exists."""
        self.assertTrue(
            self.deploy_script.exists(),
            f"deploy.sh missing at {self.deploy_script}",
        )

    def test_aws_md_exists(self):
        """Verify AWS.md documentation exists."""
        self.assertTrue(self.aws_md.exists(), "AWS.md missing")

    def test_compose_ec2_exists_and_valid_yaml(self):
        """Verify docker-compose.ec2.yml exists and is valid YAML."""
        self.assertTrue(
            self.compose_ec2.exists(), "docker-compose.ec2.yml does not exist"
        )
        with open(self.compose_ec2, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertIsInstance(data, dict)
        self.assertIn("services", data)

    # ------------------------------------------------------------------
    # Bootstrap script content
    # ------------------------------------------------------------------

    def test_bootstrap_installs_docker(self):
        """Verify bootstrap script installs Docker Engine."""
        content = self.bootstrap_script.read_text(encoding="utf-8")
        self.assertIn("docker-ce", content)
        self.assertIn("docker-compose-plugin", content)

    def test_bootstrap_enables_docker_service(self):
        """Verify bootstrap script enables and starts Docker service."""
        content = self.bootstrap_script.read_text(encoding="utf-8")
        self.assertIn("systemctl enable docker", content)
        self.assertIn("systemctl start docker", content)

    def test_bootstrap_adds_ubuntu_user_to_docker_group(self):
        """Verify bootstrap adds ubuntu user to docker group."""
        content = self.bootstrap_script.read_text(encoding="utf-8")
        self.assertIn("usermod -aG docker ubuntu", content)

    # ------------------------------------------------------------------
    # Deploy script content
    # ------------------------------------------------------------------

    def test_deploy_script_uses_ec2_ip_arg_not_localhost(self):
        """Verify deploy.sh does NOT hardcode localhost as NEXT_PUBLIC_API_BASE_URL."""
        content = self.deploy_script.read_text(encoding="utf-8")
        # Should not hard-code localhost:8000 as the backend URL
        self.assertNotIn(
            "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000", content
        )
        # Should reference the EC2 IP variable
        self.assertIn("EC2_PUBLIC_IP", content)
        self.assertIn("NEXT_PUBLIC_API_BASE_URL", content)

    def test_deploy_script_does_not_contain_real_ip(self):
        """Verify deploy.sh does not hardcode a real routable IP address."""
        content = self.deploy_script.read_text(encoding="utf-8")
        # Match any hardcoded non-placeholder public IPv4
        hardcoded_ips = re.findall(
            r"NEXT_PUBLIC_API_BASE_URL=http://(\d+\.\d+\.\d+\.\d+)",
            content,
        )
        self.assertEqual(
            hardcoded_ips,
            [],
            f"deploy.sh appears to hardcode a real IP: {hardcoded_ips}",
        )

    def test_deploy_script_runs_docker_compose_build(self):
        """Verify deploy.sh calls docker compose build."""
        content = self.deploy_script.read_text(encoding="utf-8")
        self.assertIn("docker compose build", content)

    def test_deploy_script_runs_docker_compose_up(self):
        """Verify deploy.sh calls docker compose up."""
        content = self.deploy_script.read_text(encoding="utf-8")
        self.assertIn("docker compose up -d", content)

    # ------------------------------------------------------------------
    # EC2 Compose override
    # ------------------------------------------------------------------

    def test_compose_ec2_has_restart_always(self):
        """Verify docker-compose.ec2.yml sets restart: always for EC2 reboots."""
        content = self.compose_ec2.read_text(encoding="utf-8")
        self.assertIn("restart: always", content)

    def test_compose_ec2_references_env_var_not_localhost(self):
        """Verify docker-compose.ec2.yml uses ${NEXT_PUBLIC_API_BASE_URL} not localhost."""
        content = self.compose_ec2.read_text(encoding="utf-8")
        self.assertIn("NEXT_PUBLIC_API_BASE_URL", content)
        # Should NOT default to localhost in the EC2 override
        self.assertNotIn(
            "NEXT_PUBLIC_API_BASE_URL:-http://localhost:8000", content
        )

    # ------------------------------------------------------------------
    # AWS.md documentation
    # ------------------------------------------------------------------

    def test_aws_md_documents_port_3000(self):
        """Verify AWS.md documents port 3000 security group rule."""
        content = self.aws_md.read_text(encoding="utf-8")
        self.assertIn("3000", content)

    def test_aws_md_documents_port_8000(self):
        """Verify AWS.md documents port 8000 security group rule."""
        content = self.aws_md.read_text(encoding="utf-8")
        self.assertIn("8000", content)

    def test_aws_md_documents_ec2_instance_type(self):
        """Verify AWS.md specifies an EC2 instance type."""
        content = self.aws_md.read_text(encoding="utf-8")
        self.assertIn("t3", content)

    def test_aws_md_documents_environment_variables(self):
        """Verify AWS.md documents NEXT_PUBLIC_API_BASE_URL configuration."""
        content = self.aws_md.read_text(encoding="utf-8")
        self.assertIn("NEXT_PUBLIC_API_BASE_URL", content)

    # ------------------------------------------------------------------
    # env.example documentation
    # ------------------------------------------------------------------

    def test_env_example_documents_ec2_public_ip(self):
        """Verify .env.example documents EC2_PUBLIC_IP variable."""
        content = self.env_example.read_text(encoding="utf-8")
        self.assertIn("EC2_PUBLIC_IP", content)

    # ------------------------------------------------------------------
    # Security checks
    # ------------------------------------------------------------------

    def test_no_aws_keys_in_deploy_script(self):
        """Verify deploy.sh does not contain AWS access keys."""
        content = self.deploy_script.read_text(encoding="utf-8")
        self.assertNotIn("AKIA", content)
        self.assertNotIn("aws_access_key_id", content.lower())
        self.assertNotIn("aws_secret_access_key", content.lower())

    def test_no_aws_keys_in_bootstrap_script(self):
        """Verify ec2_bootstrap.sh does not contain AWS access keys."""
        content = self.bootstrap_script.read_text(encoding="utf-8")
        self.assertNotIn("AKIA", content)
        self.assertNotIn("aws_access_key_id", content.lower())

    def test_no_aws_keys_in_env_example(self):
        """Verify .env.example does not contain real AWS credentials."""
        content = self.env_example.read_text(encoding="utf-8")
        self.assertNotIn("AKIA", content)
        self.assertNotIn("BEGIN RSA PRIVATE KEY", content)

    def test_backend_cors_allows_origins(self):
        """Verify FastAPI CORS is configured with allow_origins for remote access."""
        content = self.api_main.read_text(encoding="utf-8")
        self.assertIn("CORSMiddleware", content)
        # Wildcard or explicit origins must be present
        self.assertTrue(
            '"*"' in content or "allow_origins" in content,
            "CORS allow_origins not configured in main.py",
        )

    def test_frontend_dockerfile_accepts_build_arg(self):
        """Verify frontend Dockerfile accepts NEXT_PUBLIC_API_BASE_URL as build arg."""
        content = self.frontend_dockerfile.read_text(encoding="utf-8")
        self.assertIn("ARG NEXT_PUBLIC_API_BASE_URL", content)
        self.assertIn("ENV NEXT_PUBLIC_API_BASE_URL", content)


if __name__ == "__main__":
    unittest.main()
