import os
import requests
import subprocess
import time
import shutil
import logging
from typing import Optional
from huggingface_hub import hf_hub_download

# Configure logger for professional output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LlamaCppManager")

class LlamaCppManager:
    """
    Unified controller for llama.cpp service using a lightweight 
    pre-compiled `bin` artifacts folder with autonomous GGUF downloading.
    """
    def __init__(self,
             base_url: str = "http://localhost:11434",
             cache_dir: str = "/tmp/llama_cache",
             binary_path: str = "./bin/llama-server",
             log_path: str = "./llama.log",
             bin_artifacts_dir: str = "/content/UQ-Text-Mining/notebooks/uq_toolbox/llama_bin_artifacts",
             drive_backup_dir: str = "/content/drive/MyDrive/UQ_Toolbox_Backups"):

        self.base_url = base_url.strip().rstrip("/")
        self.port = self.base_url.split(":")[-1] if ":" in self.base_url else "11434"
        self.cache_dir = os.path.abspath(cache_dir)
        self.binary_path = os.path.abspath(binary_path)
        self.log_path = os.path.abspath(log_path)
        
        self.bin_artifacts_dir = os.path.abspath(bin_artifacts_dir)
        self.drive_backup_dir = os.path.abspath(drive_backup_dir)

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

    def _setup_prebuilt_binaries(self) -> bool:
        """
        Configures the environment using exclusively the lightweight `llama_bin_artifacts` folder 
        containing only the executable and .so libraries, restoring it from Google Drive if necessary.
        """
        work_dir = os.path.dirname(self.binary_path)
        server_binary = os.path.join(self.bin_artifacts_dir, "llama-server")

        logger.info(f"🚚 [Setup] Checking lightweight binaries in: {self.bin_artifacts_dir}...")

        # 1. If the local folder does not have the executable, try restoring from Google Drive
        if not os.path.exists(server_binary):
            logger.info("☁️ Binaries not found locally. Attempting restore from Google Drive...")
            try:
                from google.colab import drive
                if not os.path.exists("/content/drive"):
                    drive.mount("/content/drive")

                drive_zip = os.path.join(self.drive_backup_dir, "llama_bin_artifacts.zip")
                if os.path.exists(drive_zip):
                    os.makedirs(self.bin_artifacts_dir, exist_ok=True)
                    shutil.unpack_archive(drive_zip, self.bin_artifacts_dir, 'zip')
                    logger.info("✅ Artifacts successfully restored from Google Drive!")
            except Exception as e:
                logger.warning(f"⚠️ Failed to restore from Google Drive: {e}")

        # 2. If it still doesn't exist, temporarily compile from source (one-off) and isolate only the bin/lib
        server_binary = os.path.join(self.bin_artifacts_dir, "llama-server")
        if not os.path.exists(server_binary):
            logger.info("⚙️ [Compilation] No artifacts found. Performing clean compilation...")
            temp_src = os.path.join("/content", "temp_llama_build_src")
            temp_build = os.path.join(temp_src, "build")

            try:
                if os.path.exists(temp_src):
                    shutil.rmtree(temp_src)

                subprocess.run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git", temp_src], check=True)
                subprocess.run(["cmake", "-S", temp_src, "-B", temp_build, "-DGGML_CUDA=ON"], check=True)
                subprocess.run(["cmake", "--build", temp_build, "--config", "Release", "-j4"], check=True)

                # Find the compiled executable
                compiled_server = os.path.join(temp_build, "bin", "llama-server")
                if not os.path.exists(compiled_server):
                    compiled_server = os.path.join(temp_build, "llama-server")

                if not os.path.exists(compiled_server):
                    raise FileNotFoundError("Compilation completed but 'llama-server' executable not found.")

                # Create the clean lightweight binaries folder
                os.makedirs(self.bin_artifacts_dir, exist_ok=True)
                shutil.copy(compiled_server, os.path.join(self.bin_artifacts_dir, "llama-server"))

                # Collect only the .so files generated in the build
                for root, dirs, files in os.walk(temp_build):
                    for file_name in files:
                        if ".so" in file_name:
                            shutil.copy(os.path.join(root, file_name), os.path.join(self.bin_artifacts_dir, file_name))

                # Clean up heavy temporary sources
                shutil.rmtree(temp_src)
                logger.info("✨ Temporary sources removed. Only lightweight artifacts created.")

                # Save lightweight archive to Google Drive permanently
                try:
                    from google.colab import drive
                    if not os.path.exists("/content/drive"):
                        drive.mount("/content/drive")
                    os.makedirs(self.drive_backup_dir, exist_ok=True)
                    
                    zip_base = os.path.join("/content", "llama_bin_artifacts_temp")
                    shutil.make_archive(zip_base, 'zip', self.bin_artifacts_dir)
                    shutil.copy(f"{zip_base}.zip", os.path.join(self.drive_backup_dir, "llama_bin_artifacts.zip"))
                    if os.path.exists(f"{zip_base}.zip"):
                        os.remove(f"{zip_base}.zip")
                        
                    logger.info("☁️ Lightweight backup permanently saved to Google Drive.")
                except Exception as ex:
                    logger.warning(f"⚠️ Failed to save to Google Drive: {ex}")

            except Exception as e:
                logger.error(f"💥 Critical error during compilation: {e}")
                if os.path.exists(temp_src):
                    shutil.rmtree(temp_src)
                return False

        # Final verification of the executable in the lightweight binaries
        server_binary = os.path.join(self.bin_artifacts_dir, "llama-server")
        if not os.path.exists(server_binary):
            logger.error("❌ The 'llama-server' executable could not be found in the artifacts.")
            return False

        try:
            # Copy the executable to the final destination and set permissions
            if os.path.exists(self.binary_path):
                os.remove(self.binary_path)

            shutil.copy(server_binary, self.binary_path, follow_symlinks=True)
            os.chmod(self.binary_path, 0o755)

            # Copy all dynamic libraries (.so) to the server working directory
            so_count = 0
            for file_name in os.listdir(self.bin_artifacts_dir):
                if ".so" in file_name:
                    src_so = os.path.join(self.bin_artifacts_dir, file_name)
                    dst_so = os.path.join(work_dir, file_name)
                    if os.path.exists(dst_so):
                        os.remove(dst_so)
                    shutil.copy(src_so, dst_so, follow_symlinks=True)
                    so_count += 1

            logger.info(f"✅ Successfully configured the executable and {so_count} dynamic libraries.")
            return True

        except Exception as e:
            logger.error(f"💥 Critical error during binary configuration: {e}")
            return False

    def ensure_service(self, model_path: str, mmproj_path: Optional[str] = None) -> bool:
        """Ensures the server is active, adding mmproj support if provided."""
        if self.is_running():
            return True

        if not os.path.exists(self.binary_path):
            if not self._setup_prebuilt_binaries():
                return False

        env = os.environ.copy()
        bin_dir = os.path.dirname(self.binary_path)
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing_ld}" if existing_ld else bin_dir

        cmd = [
            self.binary_path,
            "-m", model_path,
        ]

        # ✨ Adds the vision projector if the model is multimodal (e.g., LLaVA)
        if mmproj_path:
            cmd.extend(["--mmproj", mmproj_path])

        cmd.extend([
            "--host", "0.0.0.0",
            "--port", self.port,
            "--n-gpu-layers", "-1",
            "--no-mmap"
        ])

        try:
            self.log_file = open(self.log_path, "w")
            self.process = subprocess.Popen(cmd, stdout=self.log_file, stderr=self.log_file, env=env)
        except Exception as e:
            logger.error(f"💥 Process startup error: {e}")
            return False

        logger.info(f"⏳ Waiting for llama-server initialization (Logging to: {self.log_path})...")

        for i in range(15):
            ret_code = self.process.poll()
            if ret_code is not None:
                logger.error(f"🛑 The llama-server subprocess DIED prematurely with exit code: {ret_code}")

                if ret_code == -9:
                    logger.error("💡 Exit code -9: SIGKILL (Kernel OOM Killer due to lack of VRAM/RAM).")
                elif ret_code == -11:
                    logger.error("💡 Exit code -11: SIGSEGV (Segmentation Fault).")
                elif ret_code != 0:
                    logger.error(f"💡 Anomalous exit code: {ret_code}")

                if os.path.exists(self.log_path):
                    with open(self.log_path, "r") as f:
                        logger.error(f"📄 Full log dump:\n{f.read()}")
                return False

            if self.is_running():
                logger.info("🎉 SUCCESS: llama-server is active.")
                return True
            time.sleep(2)

        ret_code = self.process.poll()
        logger.error(f"⏳ Timeout expired. Current process state (.poll()): {ret_code}")
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                logger.error(f"🛑 Last lines of the log:\n{f.read()[-1500:]}")

        return False

    
    def load_model(self, repo_id: str, filename: str, mmproj_filename: Optional[str] = None) -> bool:
        """
        Downloads GGUF weights and any optional mmproj file from Hugging Face,
        cleans up any previous instances, and starts the service with mmproj.
        """
        try:
            # ✨ Force close any previous server left in memory
            self._execute_pure_purge()

            logger.info(f"📥 Downloading weights from Repo: '{repo_id}' | File: '{filename}'...")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.cache_dir
            )

            mmproj_path = None
            if mmproj_filename:
                logger.info(f"📥 Downloading mmproj vision projector: '{mmproj_filename}'...")
                mmproj_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=mmproj_filename,
                    cache_dir=self.cache_dir
                )

            return self.ensure_service(model_path, mmproj_path=mmproj_path)

        except Exception as e:
            logger.error(f"❌ Download or loading failed for '{repo_id}/{filename}': {e}")
            return False
