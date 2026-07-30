import json
import boto3
import google.generativeai as genai
from botocore.exceptions import ClientError
import re
class CognitiveManager:
    def __init__(self, gemini_api_key: str, s3_bucket: str, model_name: str = "gemini-3.5-flash-lite"):
        genai.configure(api_key=gemini_api_key, transport="rest")
        self.model = genai.GenerativeModel(model_name)
        self.s3_client = boto3.client('s3')
        self.bucket = s3_bucket
        
        # Max tokens limit (~4 chars per token -> 150k tokens = 600,000 chars)
        self.MAX_CHARS_MEMORY = 120000 * 4

    def update_agent_cognition(self, task_name: str, current_action: str, execution_heads: str, execution_stderr: str) -> str:
        """
        Executes the hierarchical cognitive update. 
        Appends action + execution results to memory.txt on S3 and updates state/reports.
        """
        safe_task_name = task_name.lower().replace(' ', '-')
        cog_prefix = f"{safe_task_name}/memories"
        
        exec_data = {
            "heads": execution_heads if execution_heads else "No outputs generated.",
            "stderr": execution_stderr if execution_stderr else "No errors detected."
        }
        
        # 1. Fetch previous state directly from S3
        prev_mem = self.read_s3_text(f"{cog_prefix}/memory.txt")
        prev_state = self.read_s3_text(f"{cog_prefix}/state.txt")
        prev_report = self.read_s3_text(f"{cog_prefix}/report.txt")

        # 2. Append action & outputs to memory log (or compress if over limit)
        print("🧠 [Step 1/3] Updating Memory Log...")
        memory_result = self._handle_memory(prev_mem, exec_data, current_action)
        
        # 3. Update project management state
        print("🧠 [Step 2/3] Updating State and generating State Summary...")
        state_result = self._generate_state(prev_state, exec_data, memory_result["memory_summary"])
        
        # 4. Update long-term academic report
        print("🧠 [Step 3/3] Updating Analytical Report...")
        report_result = self._generate_report(prev_report, exec_data, memory_result["memory_summary"], state_result["state_summary"])

        # 5. Persist all 5 cognitive files back to S3
        print(f"☁️ Saving updated cognitive state to s3://{self.bucket}/{cog_prefix}/")
        self.write_s3_text(f"{cog_prefix}/memory.txt", memory_result["updated_memory"])
        self.write_s3_text(f"{cog_prefix}/memory_summary.txt", memory_result["memory_summary"])
        self.write_s3_text(f"{cog_prefix}/state.txt", state_result["updated_state"])
        self.write_s3_text(f"{cog_prefix}/state_summary.txt", state_result["state_summary"])
        self.write_s3_text(f"{cog_prefix}/report.txt", report_result["updated_report"])

        return f"s3://{self.bucket}/{cog_prefix}/"

    # ==========================================
    # MEMORY & LLM PIPELINE STEPS
    # ==========================================

    def _handle_memory(self, prev_memory: str, exec_data: dict, action: str) -> dict:
        """
        Directly appends action + results to the raw memory text.
        Only triggers LLM auto-compression if size exceeds MAX_CHARS_MEMORY.
        """
        new_entry = (
            f"\n\n==================== ITERATION ====================\n"
            f"[ACTION & CODE]:\n{action}\n\n"
            f"[EXECUTION OUTPUTS]:\n{exec_data['heads']}\n\n"
            f"[EXECUTION ERRORS]:\n{exec_data['stderr']}\n"
            f"==================================================="
        )
        
        raw_memory = (prev_memory + new_entry).strip()

        # If under threshold, return raw append log immediately (0 API calls)
        if len(raw_memory) <= self.MAX_CHARS_MEMORY:
            return {
                "updated_memory": raw_memory,
                "memory_summary": raw_memory  # Planner reads raw memory directly
            }

        # Threshold exceeded -> Compress memory via Gemini
        print("⚠️ Memory threshold exceeded. Triggering LLM auto-compression...")
        prompt = f"""You are the Memory Compression Module.
The agent's memory log has exceeded its context limit. Compress the log into a dense, highly factual narrative.
Retain all exact metrics, dataset discoveries, specific error messages, and successful code strategies.

[RAW MEMORY LOG TO COMPRESS]:
{raw_memory}

Output JSON with keys:
"updated_memory": string (the full compressed chronological log),
"memory_summary": string (a concise high-level overview for planning)."""
        
        res = self._call_gemini_json(prompt)
        
        compressed_mem = res.get("updated_memory", "")
        if len(compressed_mem) > self.MAX_CHARS_MEMORY:
            compressed_mem = "... [TRUNCATED BY SYSTEM] ...\n" + compressed_mem[-self.MAX_CHARS_MEMORY:]
            
        return {
            "updated_memory": compressed_mem, 
            "memory_summary": res.get("memory_summary", compressed_mem)
        }

    def _generate_state(self, prev_state: str, exec_data: dict, mem_summary: str) -> dict:
        prompt = f"""You are the Project Management Module tracking the operational trajectory of an autonomous agent.
### PURPOSE
Your job is to produce a dynamic, structured **Project State Document**. 
Note: Do NOT write an analytical report of conclusions (that belongs in report.txt). Instead, capture operational progress, live hypotheses, blockers, and strategic direction.

---

### INPUT CONTEXT:
[Memory Context]:
{mem_summary}

[Last Execution Outputs]:
{exec_data.get('heads', '')}

[Last Execution Errors]:
{exec_data.get('stderr', '')}

[Previous Project State Document]:
{prev_state if prev_state else "No previous state. This is the initial run."}

---

### INSTRUCTIONS FOR OUTPUT FORMATTING:

1. **`state_summary`** (1-2 sentences max):
   - A single, high-level status string for rapid logging (e.g., "Run succeeded; feature engineering boosted validation score. Preparing model training phase.").

2. **`updated_state`** (Full Markdown Document):
   - MUST be a rich, comprehensive, multi-section Markdown string. 
   - DO NOT summarize into a single line or short paragraph.
   - Synthesize the [Previous Project State] with the [Last Execution Outputs/Errors] to update all sections.
   - You MUST use the following exact structure:

```markdown
# Current Project State

## 1. Last Action Result & Status
- State whether the last run succeeded or failed.
- Key operational takeaways from execution output/logs.

## 2. Active Blockers & Risks
- Current bugs, syntax errors, missing data, or computational limits.
- Potential pitfalls for upcoming attempts.

## 3. Current Ideas & Hypotheses
- What active strategies or ideas are being tested right now?
- What hypotheses were validated/invalidated by the last run?

## 4. Cumulative Progress & Key Milestones
- High-level record of what has been accomplished so far across all runs.

## 5. Immediate Next Steps
- Concrete, actionable steps for the next iteration.

Output JSON with keys: "updated_state" (string), "state_summary" (string)."""
        
        return self._call_gemini_json(prompt)

    def _generate_report(self, prev_report: str, exec_data: dict, mem_summary: str, state_summary: str) -> dict:
        prompt = f"""You are the Analytical Reporting Module.
Append new detailed findings, data observations, validation scores, and metrics to the running report.

[Memory Context]:
{mem_summary}

[State Summary]:
{state_summary}

[Last Execution Outputs]:
{exec_data['heads']}

[Previous Report]:
{prev_report if prev_report else "No initial report."}

Output JSON with key: "updated_report" (string)."""
        
        return self._call_gemini_json(prompt)

    # ==========================================
    # PUBLIC S3 UTILITIES
    # ==========================================

    def read_s3_text(self, key: str) -> str:
        """Reads plain text from S3, returning empty string if key does not exist."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read().decode('utf-8')
        except ClientError as e:
            if e.response['Error']['Code'] in ['NoSuchKey', '404']:
                return ""
            raise e

    def write_s3_text(self, key: str, content: str):
        """Writes plain text directly to S3."""
        self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=content.encode('utf-8'))

    def _call_gemini_json(self, prompt: str) -> dict:
        """Helper to invoke Gemini with enforced JSON response structure and escape sanitization."""
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )

        raw_text = response.text

        try:
            # 1. Try standard JSON parsing
            return json.loads(raw_text, strict=False)
        except json.JSONDecodeError:
            # 2. Fix unescaped backslashes using regex
            # This replaces any single backslash NOT followed by valid JSON escape chars (", \, /, b, f, n, r, t, uXXXX)
            sanitized_text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', raw_text)
            try:
                return json.loads(sanitized_text, strict=False)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse Gemini JSON output: {e}")
                print(f"Raw Output:\n{raw_text[:500]}...")
                raise e 