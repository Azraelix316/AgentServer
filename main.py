import os
# Correctly set process environment variables
os.environ["GRPC_POLL_STRATEGY"] = "poll"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["PYTHONHTTPSVERIFY"] = "1"
import logging

# Configure root logger to output timestamps and log levels
logging.basicConfig(
    level=logging.INFO, # Change to logging.DEBUG for maximum verbosity
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Silence noisy third-party loggers unless they throw errors
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Force google genai & grpc to output internal trace logs
logging.getLogger("google").setLevel(logging.DEBUG)
logging.getLogger("grpc").setLevel(logging.DEBUG)
import time
import shutil
import boto3
import subprocess
from botocore.exceptions import ClientError

# Import your helpers
from UpdateTaskStatus import update_task_status
from PlanAgent import PlannerAgent
from CoderAgent import CoderAgent
from KaggleHelper import KaggleHelper
from EC2S3Helper import EC2OutputSyncer
from OutputLogParser import OutputLogParser
from LogAgent import CognitiveManager

class AgentTaskOrchestrator:
    def __init__(self, table_name: str = "AgentTasks"):
        print("🚀 Initializing Ephemeral Agent Task Orchestrator...")
        
        self.table_name = table_name
        
        # EC2 will automatically use its attached IAM Instance Profile.
        # No local profile or explicit keys needed for Boto3!
        self.dynamodb = boto3.resource('dynamodb', region_name=os.environ.get('AWS_REGION', 'ap-southeast-2'))
        self.table = self.dynamodb.Table(self.table_name)
        
        # Load external service credentials
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

    def run_once(self):
        """
        Executes a single pass over the DynamoDB table to process actionable tasks.
        Designed to run on a transient EC2 instance and exit when finished.
        """
        print("🔄 Starting Orchestrator Single-Pass Execution...\n")
        
        try:
            # 1. Fetch active tasks
            response = self.table.scan()
            tasks = response.get('Items', [])
            
            # Filter out tasks that are idle or finished to clean up logging
            actionable_tasks = [t for t in tasks if t.get('status') not in ['running_kaggle', 'completed', 'failed_permanently']]
            
            if not actionable_tasks:
                print("✅ No actionable tasks found in DynamoDB.")
                return

            # Process each task that needs action
            for task in actionable_tasks:
                self._process_task(task)
                
        except Exception as e:
            print(f"⚠️ Orchestrator Error: {e}")
        
        print("🏁 Single-pass execution complete.")

    def _process_task(self, task: dict):
        """
        Routes the task to the correct handler based on its status.
        """
        task_id = task.get('id')
        status = task.get('status')
        task_name = task.get('task_name', f'task-{task_id}')
        cog_prefix = f"{task_name.lower().replace(' ', '-')}/memories"

        print(f"\n⚙️ Processing Task: {task_id} | Current Status: {status}")

        try:
            # -------------------------------------------------------------
            # STATE: QUEUED -> Synchronous Plan -> Code -> Push to Kaggle
            # -------------------------------------------------------------
            if status == 'queued':
                print(f"🧠 [1/3] Fetching memory summary from S3...")
                mem_summary = self.cognition.read_s3_text(f"{cog_prefix}/memory_summary.txt")

                print(f"📋 [2/3] Generating plan...")
                plan_str = self.planner.generate_plan(
                    original_task_prompt=task.get('initial_model_prompt', task_name),
                    memory_content=mem_summary,
                    last_heads="",
                    last_stderr=""
                )
                if "TASK_COMPLETE" in plan_str:
                    print("🎯 Planner detected 'TASK_COMPLETE'! Goal achieved. Stopping task.")
                    update_task_status(
                        task_id=task_id,
                        status='completed',
                        table_name=self.table_name,
                        additional_attributes={"latest_plan": plan_str}
                    )
                    return  # Exit immediately without generating code or pushing to Kaggle
                print(f"💻 [3/3] Generating executable Python code...")
                code_str = self.coder.generate_code(
                    original_task=task.get('initial_model_prompt', task_name),
                    current_plan=plan_str
                )

                if not code_str:
                    print("❌ Code generation failed. Marking status as failed_generation.")
                    update_task_status(task_id, 'failed_generation', self.table_name)
                    return

                # Save the combined action (plan + code) directly to S3
                combined_action_log = f"PLAN:\n{plan_str}\n\nEXECUTED CODE:\n<code>\n{code_str}\n</code>"
                self.cognition.write_s3_text(f"{cog_prefix}/latest_action.txt", combined_action_log)
                # Push execution script to Kaggle
                print(f"🚀 Pushing kernel to Kaggle...")
                kernel_slug = self.kaggle.prepare_and_push(
                    task_id=task_id,
                    task_name=task_name,
                    database_link=task.get('database_link'),  # <-- Fetches database_link from DynamoDB item if present
                    agent_python_code=code_str
                )

                # Update DynamoDB
                update_task_status(
                    task_id=task_id,
                    status='running_kaggle',
                    table_name=self.table_name,
                    additional_attributes={"kernel_slug": kernel_slug}
                )

            # -------------------------------------------------------------
            # STATE: SYNC_COMPLETE -> Parse Local Files & Update Cognition
            # -------------------------------------------------------------
            elif status == 'sync_complete':
                kernel_slug = task.get('kernel_slug')
                local_dir = f"/tmp/kaggle_outputs/{kernel_slug}"

                print(f"📄 Parsing local execution outputs in {local_dir}...")
                parsed_logs = self.parser.parse_directory(local_dir)

                latest_action = self.cognition.read_s3_text(f"{cog_prefix}/latest_action.txt")
                if not latest_action:
                    latest_action = "Executed Kaggle script."

                print(f"🧠 Updating memory, state, and report artifacts in S3...")
                s3_memories_uri = self.cognition.update_agent_cognition(
                    task_name=task_name,
                    current_action=latest_action,
                    execution_heads=parsed_logs["heads"],
                    execution_stderr=parsed_logs["stderr"]
                )

                # Clean up local temporary files from EC2 disk
                if os.path.exists(local_dir):
                    shutil.rmtree(local_dir)
                # No completion keyword found -> re-queue for the next loop iteration
                print("🔄 Task ongoing. Re-queuing for next iteration...")
                update_task_status(
                    task_id=task_id,
                    status='queued', 
                    table_name=self.table_name,
                    additional_attributes={"s3_memories_uri": s3_memories_uri}
                )

            # -------------------------------------------------------------
            # STATE: KAGGLE_SUCCESS / KAGGLE_FINISHED -> Sync Local Files
            # -------------------------------------------------------------
            elif status in ['kaggle_success', 'kaggle_finished']:
                print(f"📥 Triggering EC2OutputSyncer to pull Kaggle artifacts...")
                self.syncer.process_finished_tasks()

            elif status == 'kaggle_failed':
                error_log = str(task.get('last_log', 'Kaggle execution failed with an unknown error.'))
                print(f"❌ Task '{task_id}' failed on Kaggle. Ingesting stack trace into memory...")

                latest_action = self.cognition.read_s3_text(f"{cog_prefix}/latest_action.txt")
                if not latest_action:
                    latest_action = "Executed Kaggle script."

                # Pass the Kaggle stack trace straight to CognitiveManager
                s3_memories_uri = self.cognition.update_agent_cognition(
                    task_name=task_name,
                    current_action=latest_action,
                    execution_heads="",
                    execution_stderr=error_log
                )

                # Re-queue for the next loop iteration (Planner will read updated memory_summary.txt)
                print("🔄 Error context saved to S3 memory. Re-queuing task for fix...")
                update_task_status(
                    task_id=task_id,
                    status='queued', 
                    table_name=self.table_name,
                    additional_attributes={"s3_memories_uri": s3_memories_uri}
                )

            else:
                print(f"⚠️ Unhandled status '{status}' for task {task_id}")

        except Exception as e:
            print(f"❌ Error processing task {task_id} in state '{status}': {e}")

if __name__ == "__main__":
    orchestrator = AgentTaskOrchestrator(table_name="AgentTasks")
    
    # Define runtime duration (10 minutes = 600 seconds)
    RUN_DURATION_SECONDS = 600
    POLL_INTERVAL_SECONDS = 60  # Sleep time between table scans
    
    start_time = time.time()
    print(f"⏰ Starting orchestrator continuous loop for {RUN_DURATION_SECONDS // 60} minutes...")
    
    try:
        while (time.time() - start_time) < RUN_DURATION_SECONDS:
            elapsed = int(time.time() - start_time)
            remaining = RUN_DURATION_SECONDS - elapsed
            print(f"\n⏱️ Elapsed: {elapsed}s | Remaining: {remaining}s")
            
            # Execute a pass over actionable tasks
            orchestrator.run_once()
            
            # Pause before scanning DynamoDB again
            time.sleep(POLL_INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        print("\n⚠️ Loop manually interrupted.")
    except Exception as e:
        print(f"❌ Unexpected error in main loop: {e}")

    # Shutdown safeguarding after the 10-minute window expires
    print("🛑 10-minute execution window completed. Triggering EC2 shutdown...")
    try:
        # Shutdown immediately (+0) or with a slight grace period
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
    except Exception as e:
        print(f"Failed to execute shutdown command: {e}")