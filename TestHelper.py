import os
import boto3
from EC2S3Helper import EC2OutputSyncer

# =====================================================================
# 🛠️ LOCAL DEV / TEST PROFILE CONFIGURATION
# Delete or comment out this block when deploying to EC2!
# =====================================================================
AWS_PROFILE = "test_only"
boto3.setup_default_session(profile_name=AWS_PROFILE)
print(f"🔧 Configured boto3 session using local profile: '{AWS_PROFILE}'")
# =====================================================================


def run_autonomous_syncer_test():
    TEST_BUCKET_NAME = os.environ.get("S3_BUCKET", "your-test-bucket-name")
    TEST_TABLE_NAME = os.environ.get("TABLE_NAME", "AgentTasks")
    
    # Path to local kaggle CLI binary
    LOCAL_KAGGLE_PATH = os.popen("which kaggle").read().strip() or "kaggle"

    print(f"\n🚀 Initializing EC2OutputSyncer...")
    syncer = EC2OutputSyncer(
        s3_bucket_name=TEST_BUCKET_NAME,
        table_name=TEST_TABLE_NAME,
        venv_kaggle_path=LOCAL_KAGGLE_PATH
    )

    print("\n--- Running process_finished_tasks() ---")
    synced_results = syncer.process_finished_tasks(target_status="kaggle_success")
    
    print(f"\n🎉 Test Completed!")
    print(f"Synced {len(synced_results)} tasks to S3:")
    for uri in synced_results:
        print(f"  • {uri}")

if __name__ == "__main__":
    run_autonomous_syncer_test()