import json
import boto3
import google.generativeai as genai
from botocore.exceptions import ClientError

class CognitiveManager:
    def __init__(self, gemini_api_key: str, s3_bucket: str, model_name: str = "gemini-3.5-flash-lite"):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.s3_client = boto3.client('s3')
        self.bucket = s3_bucket
        
        # Max tokens mapping (~4 chars per token)
        self.MAX_CHARS_MEMORY = 150000 * 4

    def update_agent_cognition(self, task_name: str, current_action: str, execution_heads: str, execution_stderr: str) -> str:
        """
        Executes the 3-step hierarchical cognitive update and saves to S3.
        Expects the parsed execution logs to be passed in directly.
        Returns the S3 prefix where the cognitive files are stored.
        """
        # Ensure cognitive files map to task_folder/memories
        safe_task_name = task_name.lower().replace(' ', '-')
        cog_prefix = f"{safe_task_name}/memories"
        
        # 1. Package the provided execution data
        exec_data = {
            "heads": execution_heads if execution_heads else "No outputs generated.",
            "stderr": execution_stderr if execution_stderr else "No errors detected."
        }
        
        # 2. Fetch previous state from S3 (defaults to empty strings if not found)
        prev_mem = self._read_s3_text(f"{cog_prefix}/memory.txt")
        prev_state = self._read_s3_text(f"{cog_prefix}/state.txt")
        prev_report = self._read_s3_text(f"{cog_prefix}/report.txt")

        print(f"🧠 [Step 1/3] Updating Memory and generating Memory Summary...")
        memory_result = self._generate_memory(prev_mem, exec_data, current_action)
        
        print("🧠 [Step 2/3] Updating State and generating State Summary...")
        state_result = self._generate_state(prev_state, exec_data, memory_result["memory_summary"])
        
        print("🧠 [Step 3/3] Updating Report...")
        report_result = self._generate_report(prev_report, exec_data, memory_result["memory_summary"], state_result["state_summary"])

        # 3. Save all 5 artifacts back to S3
        print(f"☁️ Saving cognitive artifacts to s3://{self.bucket}/{cog_prefix}/")
        self._write_s3_text(f"{cog_prefix}/memory.txt", memory_result["updated_memory"])
        self._write_s3_text(f"{cog_prefix}/memory_summary.txt", memory_result["memory_summary"])
        self._write_s3_text(f"{cog_prefix}/state.txt", state_result["updated_state"])
        self._write_s3_text(f"{cog_prefix}/state_summary.txt", state_result["state_summary"])
        self._write_s3_text(f"{cog_prefix}/report.txt", report_result["updated_report"])

        return f"s3://{self.bucket}/{cog_prefix}/"

    # ==========================================
    # LLM PIPELINE STEPS
    # ==========================================

    def _generate_memory(self, prev_memory: str, exec_data: dict, action: str) -> dict:
        prompt = f"""You are the Memory Module. 
Update the agent's sequential memory with the new action and its result. Then, provide a concise summary of the entire memory.
[Action]: {action}
[Outputs]: {exec_data['heads']}
[Errors]: {exec_data['stderr']}

[Previous Memory]:
{prev_memory}

Output JSON with keys: "updated_memory" (string), "memory_summary" (string)."""
        
        res = self._call_gemini_json(prompt)
        
        # Enforce token cropping on the raw memory
        mem = res.get("updated_memory", "")
        if len(mem) > self.MAX_CHARS_MEMORY:
            mem = "... [TRUNCATED] ...\n" + mem[-self.MAX_CHARS_MEMORY:]
            
        return {"updated_memory": mem, "memory_summary": res.get("memory_summary", "")}

    def _generate_state(self, prev_state: str, exec_data: dict, mem_summary: str) -> dict:
        prompt = f"""You are the Project Management Module.
Update the current project state (blockers, next steps, success/failure of last action). Then, provide a concise summary of the state.

[Memory Summary]: {mem_summary}
[Outputs]: {exec_data['heads']}
[Errors]: {exec_data['stderr']}

[Previous State]:
{prev_state}

Output JSON with keys: "updated_state" (string), "state_summary" (string)."""
        
        return self._call_gemini_json(prompt)

    def _generate_report(self, prev_report: str, exec_data: dict, mem_summary: str, state_summary: str) -> dict:
        prompt = f"""You are the Analytical Reporting Module.
Append new detailed findings, data observations, and metrics to the running academic report.

[Memory Summary]: {mem_summary}
[State Summary]: {state_summary}
[Outputs]: {exec_data['heads']}
[Errors]: {exec_data['stderr']}

[Previous Report]:
{prev_report}

Output JSON with key: "updated_report" (string)."""
        
        return self._call_gemini_json(prompt)

    # ==========================================
    # UTILITIES
    # ==========================================

    def _call_gemini_json(self, prompt: str) -> dict:
        """Wrapper to call Gemini and enforce JSON parsing."""
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)

    def _read_s3_text(self, key: str) -> str:
        """Reads text from S3, returns empty string if file doesn't exist."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read().decode('utf-8')
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return ""
            raise e

    def _write_s3_text(self, key: str, content: str):
        """Writes text directly to S3."""
        self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=content.encode('utf-8'))