import os
import json
import time
import uuid
import subprocess
from typing import Optional, List

class KaggleHelper:
    def __init__(
        self, 
        kaggle_username: str, 
        webhook_url: str,
        # venv_kaggle_path: str = "/home/ec2-user/agent_app/venv/bin/kaggle"
        venv_kaggle_path: str = "./"
    ):
        """
        :param kaggle_username: Kaggle account username.
        :param webhook_url: Lambda task manager endpoint URL.
        :param venv_kaggle_path: Absolute path to the kaggle CLI executable.
        """
        self.kaggle_username = kaggle_username.lower().strip()
        self.webhook_url = webhook_url
        self.venv_kaggle_path = venv_kaggle_path

    def prepare_and_push(
        self, 
        task_id: str, 
        agent_python_code: str, 
        database_link: Optional[str] = None,
        base_dir: str = "/tmp/kaggle_tasks"
    ) -> str:
        """
        Packages code with automated data loaders and webhooks, then pushes to Kaggle.
        
        :return: Generated unique kernel_slug (e.g. task123-1722000000-a1b2c3d4)
        """
        # 1. Generate unique kernel slug to prevent 409 collisions
        unique_suffix = str(uuid.uuid4())[:8]
        kernel_slug = f"{task_id.lower()}-{int(time.time())}-{unique_suffix}"
        kernel_id = f"{self.kaggle_username}/{kernel_slug}"
        
        task_dir = os.path.join(base_dir, kernel_slug)
        os.makedirs(task_dir, exist_ok=True)

        # 2. Build Notebook Cells (Data Ingestion + Agent Code + Webhook Callback)
        notebook_cells = []

        # --- CELL 1: Automatic Data Ingestion ---
        data_cell_code, dataset_sources = self._build_data_loader(database_link)
        if data_cell_code:
            notebook_cells.append(self._create_code_cell(data_cell_code))

        # --- CELL 2: Core Agent Python Code ---
        notebook_cells.append(self._create_code_cell(agent_python_code))

        # --- CELL 3: Automatic Webhook Callback & Telemetry ---
        webhook_code = self._build_webhook_code(task_id, kernel_id)
        notebook_cells.append(self._create_code_cell(webhook_code))

        # 3. Construct .ipynb file
        notebook_json = {
            "cells": notebook_cells,
            "metadata": {
                "language_info": {"name": "python"}
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        ipynb_path = os.path.join(task_dir, "script.ipynb")
        with open(ipynb_path, "w") as f:
            json.dump(notebook_json, f, indent=2)

        # 4. Construct kernel-metadata.json
        metadata = {
            "id": kernel_id,
            "title": kernel_slug,
            "code_file": "script.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_internet": "true",
            "dataset_sources": dataset_sources
        }

        metadata_path = os.path.join(task_dir, "kernel-metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # 5. Push via Kaggle CLI
        print(f"📤 Pushing kernel '{kernel_id}' to Kaggle...")
        result = subprocess.run(
            [self.venv_kaggle_path, "kernels", "push", "-p", task_dir],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Kaggle push failed for '{kernel_slug}': {result.stderr}")

        print(f"✅ Successfully pushed '{kernel_slug}' to Kaggle.")
        return kernel_slug

    def _build_data_loader(self, database_link: Optional[str]) -> (Optional[str], List[str]):
        """Generates prepended data loading python cells or mounts Kaggle datasets."""
        dataset_sources = []
        if not database_link:
            return None, dataset_sources

        database_link = database_link.strip()

        # Case A: Kaggle Dataset -> Native Mount
        if database_link.startswith("kaggle:"):
            ds_slug = database_link.replace("kaggle:", "").strip()
            dataset_sources.append(ds_slug)
            code = f"""# Automatically injected by KaggleHelper
import os
print("📂 Kaggle dataset mounted at /kaggle/input/{ds_slug.split('/')[-1]}")
"""
            return code, dataset_sources

        # Case B: AWS S3 Link -> Boto3 Download
        elif database_link.startswith("s3://"):
            code = f"""# Automatically injected by KaggleHelper
import boto3
import os

s3_uri = "{database_link}"
parts = s3_uri.replace("s3://", "").split("/", 1)
bucket_name, key = parts[0], parts[1]
filename = os.path.basename(key)

print(f"📥 Downloading {{s3_uri}} from S3...")
s3 = boto3.client('s3')
s3.download_file(bucket_name, key, filename)
print(f"✅ Downloaded to /kaggle/working/{{filename}}")
"""
            return code, dataset_sources

        # Case C: HTTP / Direct URL -> Requests Download
        elif database_link.startswith("http://") or database_link.startswith("https://"):
            code = f"""# Automatically injected by KaggleHelper
import urllib.request
import os

url = "{database_link}"
filename = url.split("/")[-1].split("?")[0] or "data_file"

print(f"📥 Downloading {{url}}...")
urllib.request.urlretrieve(url, filename)
print(f"✅ Downloaded to /kaggle/working/{{filename}}")
"""
            return code, dataset_sources

        return None, dataset_sources

    def _build_webhook_code(self, task_id: str, kernel_id: str) -> str:
        """Generates callback code injected at the end of every notebook."""
        return f"""# Automatically injected by KaggleHelper
import json
import urllib.request

WEBHOOK_URL = "{self.webhook_url}"
TASK_ID = "{task_id}"
KERNEL_ID = "{kernel_id}"

def send_callback(status, log_message):
    payload = {{
        "task_id": TASK_ID,
        "status": status,
        "last_kernel_id": KERNEL_ID,
        "last_log": log_message[:2000]  # Truncate to prevent payload explosion
    }}
    
    headers = {{"Content-Type": "application/json"}}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(WEBHOOK_URL, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            print(f"🔔 Callback sent successfully! Status: {{response.status}}")
    except Exception as e:
        print(f"❌ Callback failed to send: {{e}}")

# Report success upon reaching the end of the script execution
send_callback("kaggle_success", "Execution completed without uncaught exceptions.")
"""

    @staticmethod
    def _create_code_cell(source_code: str) -> dict:
        """Wraps a python string into a standard Jupyter Notebook cell."""
        lines = [line + "\n" for line in source_code.strip().split("\n")]
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        }