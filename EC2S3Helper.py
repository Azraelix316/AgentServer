import os
import subprocess
import boto3
from typing import List

class EC2OutputSyncer:
    def __init__(
        self, 
        s3_bucket_name: str, 
        table_name: str = "AgentTasks",
        venv_kaggle_path: str = ".venv/bin/kaggle"
    ):
        self.s3_bucket = s3_bucket_name
        self.venv_kaggle_path = venv_kaggle_path
        self.s3_client = boto3.client('s3')
        
        # DynamoDB setup
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)

    def process_finished_tasks(self, target_status: str = "kaggle_success") -> List[str]:
        """
        Scans DynamoDB for tasks with status 'kaggle_success', pulls their outputs 
        via Kaggle CLI, uploads them to S3, and marks tasks as 'completed'.
        """
        print(f"🔍 Checking DynamoDB for tasks with status '{target_status}'...")
        
        response = self.table.scan(
            FilterExpression="#st = :val",
            ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues={":val": target_status}
        )
        
        tasks = response.get('Items', [])
        if not tasks:
            print("ℹ️ No finished tasks found pending output sync.")
            return []

        synced_uris = []
        for task in tasks:
            task_id = task.get('id')
            task_name = task.get('task_name', 'default-task')
            kernel_id = task.get('last_kernel_id')  # Format: "username/kernel-slug-123"

            if not kernel_id:
                print(f"⚠️ Task '{task_id}' missing 'last_kernel_id'. Skipping...")
                continue

            # Extract the unique kernel slug directly from last_kernel_id
            kernel_slug = kernel_id.split('/')[-1] if '/' in kernel_id else kernel_id

            print(f"\n⚡ Processing Task ID: {task_id} [{task_name}]")
            
            try:
                # 1. Fetch from Kaggle and upload to S3
                s3_uri = self.fetch_and_upload_to_s3(
                    kernel_id=kernel_id, 
                    task_name=task_name, 
                    run_slug=kernel_slug
                )
                
                # 2. Update DynamoDB status to 'completed' & attach output location
                self.table.update_item(
                    Key={'id': task_id},
                    UpdateExpression="SET #st = :comp, s3_output_uri = :uri",
                    ExpressionAttributeNames={"#st": "status"},
                    ExpressionAttributeValues={
                        ":comp": "queued",
                        ":uri": s3_uri
                    }
                )
                print(f"✅ Task '{task_id}' updated to 'completed' in DynamoDB.")
                synced_uris.append(s3_uri)

            except Exception as e:
                print(f"❌ Failed to sync task '{task_id}': {e}")
                self.table.update_item(
                    Key={'id': task_id},
                    UpdateExpression="SET #st = :err, last_log = :msg",
                    ExpressionAttributeNames={"#st": "status"},
                    ExpressionAttributeValues={
                        ":err": "sync_error",
                        ":msg": str(e)
                    }
                )

        return synced_uris

    def fetch_and_upload_to_s3(self, kernel_id: str, task_name: str, run_slug: str) -> str:
        """
        1. Downloads output files from Kaggle kernel via CLI.
        2. Uploads them to s3://<bucket>/<task_name>/<run_slug>/
        """
        local_download_dir = f"/tmp/kaggle_outputs/{run_slug}"
        os.makedirs(local_download_dir, exist_ok=True)

        print(f"📥 Pulling Kaggle outputs for '{kernel_id}'...")
        res = subprocess.run(
            [self.venv_kaggle_path, "kernels", "output", kernel_id, "-p", local_download_dir],
            capture_output=True,
            text=True
        )

        if res.returncode != 0:
            print(f"⚠️ Warning pulling kernel outputs: {res.stderr}")

        # Path format: s3://<bucket>/<clean-task-name>/<kernel-slug>/<file>
        safe_task_name = task_name.lower().replace(' ', '-')
        s3_prefix = f"{safe_task_name}/{run_slug}"

        for root, _, files in os.walk(local_download_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_key = f"{s3_prefix}/{file}"
                
                print(f"☁️ Uploading {file} to s3://{self.s3_bucket}/{s3_key}")
                self.s3_client.upload_file(local_file_path, self.s3_bucket, s3_key)

        s3_uri = f"s3://{self.s3_bucket}/{s3_prefix}/"
        print(f"✅ All artifacts synced to {s3_uri}")
        return s3_uri