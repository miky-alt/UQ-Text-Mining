import os
import requests
import subprocess
import time
import shutil
import logging
from huggingface_hub import hf_hub_download

# Configure logger for professional output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LlamaCppManager")

class LlamaCppManager:
    """
    Unified controller for llama.cpp service lifecycle with autonomous 
    GGUF weight downloading via explicit Hugging Face repo_id and filename.
    """
    def __init__(self, 
                 base_url: str = "http://localhost:11434", 
                 cache_dir: str = "/tmp/llama_cache",
                 binary_path: str = "./bin/llama-server",
                 log_path: str = "./llama.log",
                 binary_url: str = "https://github.com/ggml-org/llama.cpp/releases/download/b10069/llama-b10069-bin-ubuntu-x64.tar.gz"):
        
        # Configuration attributes
        self.base_url = base_url.strip().rstrip("/")
        self.port = self.base_url.split(":")[-1] if ":" in self.base_url else "11434"
        self.cache_dir = os.path.abspath(cache_dir)
        self.binary_url = binary_url
        self.binary_path = os.path.abspath(binary_path)
        self.log_path = os.path.abspath(log_path)
        
        # Ensure cache and binary directories exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.binary_path), exist_ok=True)

    def is_running(self) -> bool:
        """Checks if the llama-server is responsive on the expected endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _execute_pure_purge(self) -> None:
        """Forces a cleanup of stale llama-server processes."""
        logger.info("🧹 [Self-Healing] Purging legacy llama-server artifacts...")
        subprocess.run(["pkill", "-9", "llama-server"], capture_output=True)

    def _attempt_system_install(self) -> bool:
        """Downloads, extracts, and deploys the binary with recursive file discovery."""
        work_dir = os.path.dirname(self.binary_path)
        temp_tar = os.path.join(work_dir, "llama_bin.tar.gz")
        extract_root = os.path.join(work_dir, "temp_extract")

        logger.info(f"📥 Fetching binary from: {self.binary_url}")
        
        try:
            # 1. Download the archive
            subprocess.run(["curl", "-L", self.binary_url, "-o", temp_tar], check=True)
            
            # 2. Extract content
            if os.path.exists(extract_root):
                shutil.rmtree(extract_root)
            os.makedirs(extract_root)
            subprocess.run(["tar", "-xzf", temp_tar, "-C", extract_root], check=True)
            
            # 3. Locate and move the binary
            found = False
            for root, dirs, files in os.walk(extract_root):
                if "llama-server" in files:
                    src_path = os.path.join(root, "llama-server")
                    shutil.move(src_path, self.binary_path)
                    found = True
                    break
            
            # 4. Cleanup temporary artifacts
            if os.path.exists(temp_tar):
                os.remove(temp_tar)
            if os.path.exists(extract_root):
                shutil.rmtree(extract_root)
            
            if not found:
                raise FileNotFoundError("Binary 'llama-server' not found in the downloaded archive.")

            os.chmod(self.binary_path, 0o755)
            logger.info("✅ Binary successfully deployed.")
            return True
        except Exception as e:
            logger.error(f"💥 Deployment failed: {e}")
            return False

    def ensure_service(self, model_path: str) -> bool:
        """Ensures the server is active, auto-restarting if necessary."""
        if self.is_running():
            return True

        if not os.path.exists(self.binary_path):
            if not self._attempt_system_install():
                return False

        # Command constructed as a list to prevent shell injection
        cmd = [
            self.binary_path, 
            "-m", model_path, 
            "--host", "0.0.0.0", 
            "--port", self.port, 
            "--n-gpu-layers", "-1"
        ]
        
        try:
            log_file = open(self.log_path, "w")
            subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        except Exception as e:
            logger.error(f"💥 Failed to launch process: {e}")
            return False
        
        logger.info(f"⏳ Waiting for llama-server initialization (Logging to: {self.log_path})...")
        for i in range(15):
            if self.is_running():
                logger.info("🎉 SUCCESS: llama-server is active.")
                return True
            time.sleep(2)
        return False

    def load_model(self, repo_id: str, filename: str) -> bool:
        """
        Downloads GGUF weights via Hugging Face using explicit repo_id and filename,
        then boots the service.
        """
        try:
            logger.info(f"📥 Autonomous acquisition from Repo: '{repo_id}' | File: '{filename}'...")
            
            # Download automatico gestito da huggingface_hub direttamente nella cache
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"📦 Model weights ready at: {model_path}")
            return self.ensure_service(model_path)

        except Exception as e:
            logger.error(f"❌ Autonomous model download/load failed for '{repo_id}/{filename}': {e}")
            return False
