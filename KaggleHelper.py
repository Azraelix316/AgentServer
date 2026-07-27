import os
import json
import time
import uuid
import subprocess
import boto3
from typing import Optional, List

class KaggleHelper:
    def __init__(
        self, 
        kaggle_username: str, 
        webhook_url: str,
        table_name: str = "AgentTasks",
        venv_kaggle_path: str = "/home/ec2-user/AgentServer/venv/bin/kaggle"
    ):
        """
        :param kaggle_username: Kaggle account username.
        :param webhook_url: Lambda task manager endpoint URL.
        :param table_name: DynamoDB table name to update task state directly.
        :param venv_kaggle_path: Executable path for Kaggle CLI.
        """
        self.kaggle_username = kaggle_username.lower().strip()
        self.webhook_url = webhook_url
        self.venv_kaggle_path = venv_kaggle_path
        
        # Connect directly to DynamoDB to manage state & error handling
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)
    def prepare_and_push(
        self, 
        task_id: str, 
        task_name: str,
        agent_python_code: str, 
        database_link: Optional[str] = None,
        base_dir: str = "/tmp/kaggle_tasks"
    ) -> str:
        """
        Packages code with task_name grouping, automated data loaders, and webhooks.
        Manages task state transitions in DynamoDB directly.
        """
        # 1. Generate the exact title string (e.g. "task-101-1785132775-c7968374")
        safe_task_name = task_name.lower().replace(" ", "-")
        unique_suffix = str(uuid.uuid4())[:8]
        
        # TITLE IS THE SLUG SOURCE OF TRUTH
        title_slug = f"{safe_task_name}-{int(time.time())}-{unique_suffix}"
        
        # Sanitize username (remove any spaces)
        clean_username = self.kaggle_username.replace(" ", "")
        
        # Construct kernel_id where suffix after / MUST match title_slug
        kernel_id = f"{clean_username}/{title_slug}"
        
        task_dir = os.path.join(base_dir, title_slug)
        os.makedirs(task_dir, exist_ok=True)

        try:
            # 2. Build Notebook Cells
            notebook_cells = []

            # CELL 1: Automatic Data Ingestion
            data_cell_code, dataset_sources = self._build_data_loader(database_link)
            if data_cell_code:
                notebook_cells.append(self._create_code_cell(data_cell_code))

            # CELL 2: Core Agent Python Code
            notebook_cells.append(self._create_code_cell(agent_python_code))

            # CELL 3: Automatic Webhook Callback
            webhook_code = self._build_webhook_code(task_id, task_name, kernel_id)
            notebook_cells.append(self._create_code_cell(webhook_code))

            # 3. Construct .ipynb file with kernelspec
            notebook_json = {
                "cells": notebook_cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "name": "python",
                        "version": "3.10.0"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }

            ipynb_path = os.path.join(task_dir, "script.ipynb")
            with open(ipynb_path, "w") as f:
                json.dump(notebook_json, f, indent=2)

            # 4. Construct kernel-metadata.json
            # Both 'id' suffix and 'title' are explicitly forced to match
            metadata = {
                "id": kernel_id,
                "title": title_slug,  # Title is now identical to the slug suffix
                "code_file": "script.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": "true",
                "enable_internet": "true",
                "dataset_sources": dataset_sources,
                "keywords": [safe_task_name, task_id.lower()]
            }

            metadata_path = os.path.join(task_dir, "kernel-metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # 5. Push via Kaggle CLI
            print(f"📤 Pushing kernel '{kernel_id}' under title '{title_slug}'...")
            result = subprocess.run(
                [self.venv_kaggle_path, "kernels", "push", "-p", task_dir],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Kaggle CLI Push Error: {result.stderr}")

            print(f"✅ Successfully pushed '{title_slug}' to Kaggle.")

            # 6. Update DynamoDB using the exact kernel_id
            self.table.update_item(
                Key={'id': task_id},
                UpdateExpression="SET #st = :status, last_kernel_id = :kid",
                ExpressionAttributeNames={"#st": "status"},
                ExpressionAttributeValues={
                    ":status": "running_kaggle",
                    ":kid": kernel_id
                }
            )
            print(f"📌 Task '{task_id}' state updated to 'running_kaggle' in DynamoDB.")
            return title_slug

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error handling triggered in KaggleHelper for task '{task_id}': {error_msg}")
            
            try:
                self.table.update_item(
                    Key={'id': task_id},
                    UpdateExpression="SET #st = :status, last_log = :err",
                    ExpressionAttributeNames={"#st": "status"},
                    ExpressionAttributeValues={
                        ":status": "push_failed",
                        ":err": error_msg[:2000]
                    }
                )
                print(f"📌 Task '{task_id}' state updated to 'push_failed' in DynamoDB.")
            except Exception as db_err:
                print(f"❌ Failed to update DynamoDB error state: {db_err}")

            raise e

    def _build_data_loader(self, database_link: Optional[str]) -> (Optional[str], List[str]):
        dataset_sources = []
        if not database_link:
            return None, dataset_sources

        database_link = database_link.strip()

        if database_link.startswith("kaggle:"):
            ds_slug = database_link.replace("kaggle:", "").strip()
            dataset_sources.append(ds_slug)
            code = f"""# Automatically injected by KaggleHelper
import os
print("📂 Kaggle dataset mounted at /kaggle/input/{ds_slug.split('/')[-1]}")
"""
            return code, dataset_sources

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

    def _build_webhook_code(self, task_id: str, task_name: str, kernel_id: str) -> str:
        return f"""# Automatically injected by KaggleHelper
import json
import urllib.request

WEBHOOK_URL = "{self.webhook_url}"
TASK_ID = "{task_id}"
TASK_NAME = "{task_name}"
KERNEL_ID = "{kernel_id}"

def send_callback(status, log_message):
    payload = {{
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "status": status,
        "last_kernel_id": KERNEL_ID,
        "last_log": log_message[:2000]
    }}
    
    headers = {{"Content-Type": "application/json"}}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(WEBHOOK_URL, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            print(f"🔔 Callback sent successfully! Status: {{response.status}}")
    except Exception as e:
        print(f"❌ Callback failed to send: {{e}}")

send_callback("kaggle_success", "Execution completed without uncaught exceptions.")
"""

    @staticmethod
    def _create_code_cell(source_code: str) -> dict:
        lines = [line + "\n" for line in source_code.strip().split("\n")]
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        }