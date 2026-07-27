import os
import subprocess
import boto3
import traceback
from typing import List

class EC2OutputSyncer:
    def __init__(
        self, 
        s3_bucket_name: str, 
        table_name: str = "AgentTasks",
        venv_kaggle_path: str = "/home/ec2-user/AgentServer/venv/bin/kaggle"
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
        via Kaggle CLI, uploads them to S3, and marks tasks as 'sync_complete'.
        """
        print(f"🔍 Checking DynamoDB for tasks with status '{target_status}'...")
        
        try:
            response = self.table.scan(
                FilterExpression="#st = :val",
                ExpressionAttributeNames={"#st": "status"},
                ExpressionAttributeValues={":val": target_status}
            )
        except Exception as e:
            print(f"❌ Failed to scan DynamoDB: {e}")
            return []
        
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
                
                # 2. Update DynamoDB status to indicate this step of the loop is done
                # 'sync_complete' tells the agent it can now evaluate the S3 files
                next_status = "sync_complete" 
                
                self.table.update_item(
                    Key={'id': task_id},
                    UpdateExpression="SET #st = :comp, s3_output_uri = :uri",
                    ExpressionAttributeNames={"#st": "status"},
                    ExpressionAttributeValues={
                        ":comp": next_status,
                        ":uri": s3_uri
                    }
                )
                print(f"✅ Task '{task_id}' updated to '{next_status}' in DynamoDB.")
                synced_uris.append(s3_uri)

            except Exception as e:
                error_trace = traceback.format_exc()
                print(f"❌ Failed to sync task '{task_id}':\n{error_trace}")
                
                try:
                    self.table.update_item(
                        Key={'id': task_id},
                        UpdateExpression="SET #st = :err, last_log = :msg",
                        ExpressionAttributeNames={"#st": "status"},
                        ExpressionAttributeValues={
                            ":err": "sync_error",
                            ":msg": error_trace[:2000]  # Safe truncation for DynamoDB
                        }
                    )
                except Exception as db_err:
                    print(f"❌ Critical failure updating error state to DynamoDB: {db_err}")

        return synced_uris

    def fetch_and_upload_to_s3(self, kernel_id: str, task_name: str, run_slug: str) -> str:
        """
        1. Downloads output files from Kaggle kernel via CLI.
        2. Uploads them to s3://<bucket>/<task_name>/<run_slug>/ maintaining folder structure.
        """
        local_download_dir = f"/tmp/kaggle_outputs/{run_slug}"
        os.makedirs(local_download_dir, exist_ok=True)

        print(f"📥 Pulling Kaggle outputs for '{kernel_id}' into '{local_download_dir}'...")
        res = subprocess.run(
            [self.venv_kaggle_path, "kernels", "output", kernel_id, "-p", local_download_dir],
            capture_output=True,
            text=True
        )

        # Detailed CLI Error Catching
        if res.returncode != 0:
            cli_error = f"Kaggle CLI kernels output failed.\nReturn Code: {res.returncode}\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
            raise RuntimeError(cli_error)

        safe_task_name = task_name.lower().replace(' ', '-')
        s3_prefix = f"{safe_task_name}/{run_slug}"

        uploaded_count = 0
        for root, _, files in os.walk(local_download_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                
                # PRESERVES FOLDER STRUCTURE (e.g. outputs/plots/chart.png)
                rel_path = os.path.relpath(local_file_path, local_download_dir)
                s3_key = f"{s3_prefix}/{rel_path}".replace("\\", "/") 
                
                print(f"☁️ Uploading {rel_path} to s3://{self.s3_bucket}/{s3_key}")
                self.s3_client.upload_file(local_file_path, self.s3_bucket, s3_key)
                uploaded_count += 1

        if uploaded_count == 0:
            raise RuntimeError(f"No output files were downloaded from Kaggle for kernel '{kernel_id}'. Did the script save files to /kaggle/working/?")

        s3_uri = f"s3://{self.s3_bucket}/{s3_prefix}/"
        print(f"✅ Synced {uploaded_count} artifacts to {s3_uri}")
        return s3_uri