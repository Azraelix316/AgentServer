import os
import shutil
from OutputLogSummarizer import OutputLogSummarizer

def create_mock_output_dir(base_dir: str):
    """Creates a temporary folder with mock outputs to test parsing logic."""
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    os.makedirs(os.path.join(base_dir, "subfolder"), exist_ok=True)

    # 1. Create a mock execution log file (contains stdout and stderr)
    log_content = """2026-07-27 10:00:00 [INFO] Script starting...
2026-07-27 10:00:01 [INFO] Loading dataset...
2026-07-27 10:00:02 [STDERR] Warning: UserWarning: Feature names only available when fit with feature_names
2026-07-27 10:00:03 [INFO] Model training finished. Accuracy: 88.5%
2026-07-27 10:00:04 [STDERR] ERROR: Failed to save model checkpoint to /invalid_path/model.pt
2026-07-27 10:00:05 [INFO] Generating metrics.csv...
"""
    with open(os.path.join(base_dir, "execution.log"), "w", encoding="utf-8") as f:
        f.write(log_content)

    # 2. Create a mock CSV output file
    csv_content = "epoch,loss,accuracy\n1,0.45,0.72\n2,0.30,0.81\n3,0.18,0.88\n"
    with open(os.path.join(base_dir, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write(csv_content)

    # 3. Create a nested mock output file inside a subfolder
    sub_csv_content = "id,prediction\n101,0.92\n102,0.14\n"
    with open(os.path.join(base_dir, "subfolder", "predictions.csv"), "w", encoding="utf-8") as f:
        f.write(sub_csv_content)

    # 4. Create a mock non-UTF8 / binary file (e.g., plot image or model checkpoint)
    with open(os.path.join(base_dir, "chart.png"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00")

    print(f"📁 Created mock test artifacts in '{base_dir}'")

def run_test():
    # Retrieve Gemini API key from environment variable or replace directly
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set it via: export GEMINI_API_KEY='your-api-key'")
        return

    test_dir = "/tmp/test_kaggle_outputs"
    
    try:
        # Step 1: Generate dummy files
        create_mock_output_dir(test_dir)

        # Step 2: Initialize Summarizer
        summarizer = OutputLogSummarizer(gemini_api_key=api_key)

        # Step 3: Run Summarization
        print("\n🚀 Testing OutputLogSummarizer...")
        summary = summarizer.generate_summary(local_output_dir=test_dir, max_head_lines=10)

        # Step 4: Display Output
        print("\n=========================================")
        print("🤖 GEMINI OUTPUT SUMMARY RESULT:")
        print("=========================================")
        print(summary)
        print("=========================================\n")

    finally:
        # Cleanup temporary files after test
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print("🧹 Cleaned up temporary test directory.")

if __name__ == "__main__":
    run_test()