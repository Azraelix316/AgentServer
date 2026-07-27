import os
import time
import boto3
from botocore.exceptions import ClientError

# Import your helpers
from UpdateTaskStatus import update_task_status
from PlanAgent import PlannerAgent
from CoderAgent import CoderAgent
from KaggleHelper import KaggleHelper
from EC2S3Helper import EC2OutputSyncer
from OutputLogParser import OutputLogParser
from LogAgent import CognitiveManager
AWS_PROFILE = "test_only"
boto3.setup_default_session(profile_name=AWS_PROFILE)
print(f"🔧 Configured boto3 session using local profile: '{AWS_PROFILE}'")
class AgentTaskOrchestrator:
    def __init__(self, table_name: str = "AgentTasks"):
        print("🚀 Initializing Agent Task Orchestrator...")
        
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(self.table_name)
        
        # Load credentials
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.kaggle_user = os.environ.get("KAGGLE_USERNAME")
        self.s3_bucket = os.environ.get("S3_BUCKET_NAME")
        self.webhook_url = os.environ.get("WEBHOOK_URL", "")
        if not all([self.gemini_key, self.kaggle_user, self.s3_bucket]):
            raise ValueError("❌ Missing required environment variables (GEMINI_API_KEY, KAGGLE_USERNAME, S3_BUCKET_NAME).")

        # Initialize Helpers
        self.planner = PlannerAgent(gemini_api_key=self.gemini_key)
        self.coder = CoderAgent(gemini_api_key=self.gemini_key)
        self.kaggle = KaggleHelper(kaggle_username=self.kaggle_user, webhook_url=self.webhook_url)
        self.syncer = EC2OutputSyncer(s3_bucket_name=self.s3_bucket)
        self.parser = OutputLogParser(max_head_lines=50)
        self.cognition = CognitiveManager(gemini_api_key=self.gemini_key, s3_bucket=self.s3_bucket)

    def run_loop(self, poll_interval_seconds: int = 10):
        """
        The main daemon loop. Continuously scans DynamoDB and processes tasks.
        """
        print(f"🔄 Starting Orchestrator Loop. Polling every {poll_interval_seconds} seconds...\n")
        
        while True:
            try:
                # 1. Fetch active tasks (scanning for simplicity; use a GSI for scale)
                response = self.table.scan()
                tasks = response.get('Items', [])
                
                for task in tasks:
                    self._process_task(task)
                    
            except Exception as e:
                print(f"⚠️ Orchestrator Error: {e}")
            
            # Sleep before the next sweep
            time.sleep(poll_interval_seconds)

    def _process_task(self, task: dict):
        """
        Routes the task to the correct handler based on its status.
        """
        task_id = task.get('id')
        status = task.get('status')
        task_name = task.get('task_name', f'task-{task_id}')
        
        # Skip tasks that are waiting on external systems or are permanently done
        if status in ['running_kaggle', 'completed', 'failed_permanently']:
            return

        print(f"\n⚙️ Processing Task: {task_id} | Current Status: {status}")

        try:
            # -------------------------------------------------------------
            # STATE: QUEUED -> Needs a Plan
            # -------------------------------------------------------------
            if status == 'queued':
                # Fetch memory summary & last logs from S3 (or default to empty if new)
                cog_prefix = f"{task_name.lower().replace(' ', '-')}/memories"
                mem_summary = self.cognition._read_s3_text(f"{cog_prefix}/memory_summary.txt")
                
                # We pull last heads/stderr from Dynamo if available, otherwise empty
                last_heads = task.get('last_heads', '')
                last_stderr = task.get('last_stderr', '')
                
                plan_str = self.planner.generate_plan(
                    original_task_prompt=task.get('original_prompt', task_name),
                    memory_content=mem_summary,
                    last_heads=last_heads,
                    last_stderr=last_stderr
                )
                
                update_task_status(task_id, 'planning_complete', self.table_name, {"current_plan": plan_str})

            # -------------------------------------------------------------
            # STATE: PLANNING_COMPLETE -> Needs Code
            # -------------------------------------------------------------
            elif status == 'planning_complete':
                code_str = self.coder.generate_code(
                    original_task=task_name,
                    current_plan=task.get('current_plan', '')
                )
                
                if code_str:
                    update_task_status(task_id, 'coding_complete', self.table_name, {"agent_code": code_str})
                else:
                    update_task_status(task_id, 'failed_generation', self.table_name)

            # -------------------------------------------------------------
            # STATE: CODING_COMPLETE -> Push to Kaggle
            # -------------------------------------------------------------
            elif status == 'coding_complete':
                kernel_slug = self.kaggle.prepare_and_push(
                    task_id=task_id,
                    task_name=task_name,
                    agent_python_code=task.get('agent_code', '')
                )
                
                update_task_status(task_id, 'running_kaggle', self.table_name, {"kernel_slug": kernel_slug})

            # -------------------------------------------------------------
            # STATE: KAGGLE_FINISHED -> Sync, Parse, and Update Cognition
            # -------------------------------------------------------------
            # (Assuming a webhook or external poller sets status to 'kaggle_finished')
            # -------------------------------------------------------------
            # STATE: KAGGLE_FINISHED -> Sync and Parse Local Files
            # -------------------------------------------------------------
            elif status == 'kaggle_success':
                kernel_slug = task.get('kernel_slug')
                if not kernel_slug:
                    print(f"❌ Missing kernel_slug for task {task_id}")
                    update_task_status(task_id, 'failed_permanently', self.table_name)
                    return

                # 1. Sync files from S3/Kaggle to EC2
                local_dir = f"/tmp/kaggle_outputs/{kernel_slug}"
                s3_outputs_uri = self.syncer.process_finished_tasks(kernel_slug, task_name, local_dir)

                # 2. Parse the local files into heads & stderr
                parsed_logs = self.parser.parse_directory(local_dir)

                # 3. Advance to sync_complete
                update_task_status(
                    task_id=task_id,
                    status='sync_complete', 
                    table_name=self.table_name,
                    additional_attributes={
                        "s3_output_uri": s3_outputs_uri,
                        "last_heads": parsed_logs["heads"],
                        "last_stderr": parsed_logs["stderr"],
                        "local_output_dir": local_dir
                    }
                )

            # -------------------------------------------------------------
            # STATE: SYNC_COMPLETE -> Run Cognitive Manager
            # -------------------------------------------------------------
            elif status == 'sync_complete':
                # Run Cognitive Manager using the parsed logs stored in DynamoDB
                s3_memories_uri = self.cognition.update_agent_cognition(
                    task_name=task_name,
                    current_action=task.get('current_plan', 'Executed Kaggle code'),
                    execution_heads=task.get('last_heads', ''),
                    execution_stderr=task.get('last_stderr', '')
                )

                # Clean up local temporary files from disk
                local_dir = task.get('local_output_dir', f"/tmp/kaggle_outputs/{task.get('kernel_slug')}")
                if os.path.exists(local_dir):
                    import shutil
                    shutil.rmtree(local_dir)

                # Re-queue the task for the Planner to start the next iteration
                update_task_status(
                    task_id=task_id,
                    status='queued', 
                    table_name=self.table_name,
                    additional_attributes={
                        "s3_memories_uri": s3_memories_uri
                    }
                )
            else:
                print(f"⚠️ Unhandled status '{status}' for task {task_id}")

        except Exception as e:
            print(f"❌ Error processing task {task_id} in state '{status}': {e}")
            # Optional: Implement a retry counter here before setting to failed_permanently

if __name__ == "__main__":
    # Ensure you have your environment variables set!
    # export GEMINI_API_KEY="your_key"
    # export KAGGLE_USERNAME="your_username"
    # export S3_BUCKET_NAME="researchagentstorage"
    
    orchestrator = AgentTaskOrchestrator(table_name="AgentTasks")
    orchestrator.run_loop(poll_interval_seconds=10)