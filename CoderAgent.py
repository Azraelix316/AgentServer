import re
import traceback
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError

class CoderAgent:
    def __init__(self, gemini_api_key: str):
        """
        Initializes the Coder Agent with a prioritized model fallback queue.
        """
        genai.configure(api_key=gemini_api_key, transport="rest")
        
        # Priority fallback queue for rate-limit / availability resilience
        self.model_queue = [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite"
        ]

    def generate_code(self, original_task: str, current_plan: str, max_retries: int = 3) -> str:
        """
        Takes the planner's checklist and generates the corresponding Python script.
        Validates syntax locally using compile(), self-corrects up to max_retries,
        and fails over through the model queue on API rate limits.
        """
        print("💻 Coder Agent is writing the Python script based on the current plan...")
        
        prompt = self._build_coder_prompt(original_task, current_plan)
        last_attempted_code = ""

        for attempt in range(1, max_retries + 1):
            # 1. Execute LLM call with fallback across models
            raw_output = self._call_gemini_with_fallback(prompt)
            
            if not raw_output:
                print("❌ Coder Agent failed to receive a response from any Gemini model.")
                return last_attempted_code

            # 2. Extract cleanly formatted code from response
            clean_code = self._extract_code(raw_output)
            
            if not clean_code:
                print(f"⚠️ Pre-flight check failed (Attempt {attempt}/{max_retries}): Missing <code> tags.")
                prompt += "\n\nCRITICAL ERROR: You failed to wrap your Python code inside <code>...</code> XML tags. Please rewrite the entire script wrapped in <code> tags."
                continue

            last_attempted_code = clean_code

            # 3. Local syntax validation via compile()
            try:
                compile(clean_code, filename="<agent_generated_code>", mode="exec")
                print(f"✅ Code generated and syntactically validated (Attempt {attempt}/{max_retries}).")
                return clean_code

            except SyntaxError as syntax_err:
                print(f"⚠️ Pre-flight SyntaxError detected (Attempt {attempt}/{max_retries}): {syntax_err}")
                
                # Feed exact line numbers and offending code back to LLM for target fixing
                prompt += (
                    f"\n\n--- PREVIOUS ATTEMPT FAILED PRE-FLIGHT SYNTAX COMPILATION ---"
                    f"\nYour previous attempt produced a Python SyntaxError:"
                    f"\nError Message: {syntax_err.msg}"
                    f"\nLine {syntax_err.lineno}: {syntax_err.text}"
                    f"\n\nOffending Code Attempt:\n<code>\n{clean_code}\n</code>"
                    f"\n\nPlease rewrite the COMPLETE script, fixing the syntax issue on or near line {syntax_err.lineno}."
                    f" Ensure all string quotes are correctly closed and all multiline strings use triple-quotes (\"\"\")."
                    f" Wrap your fixed code in <code>...</code> tags."
                )

        print("❌ Coder Agent failed to produce syntactically valid code after maximum retries.")
        return last_attempted_code

    def _call_gemini_with_fallback(self, prompt: str) -> str:
        """
        Sequentially tries models in self.model_queue.
        Switches to the next model if a 429 (ResourceExhausted) or API error occurs.
        """
        for model_name in self.model_queue:
            try:
                print(f"🤖 Invoking LLM: {model_name}...")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                
                if response and response.text:
                    return response.text

            except ResourceExhausted:
                print(f"🚨 Rate limit (429) hit for '{model_name}'. Automatically failing over to next model...")
                continue
                
            except GoogleAPICallError as e:
                print(f"⚠️ API call error on '{model_name}': {e}. Failing over to next model...")
                continue
                
            except Exception as e:
                print(f"❌ Unexpected error calling model '{model_name}': {e}")
                continue

        print("❌ All model fallbacks in queue were exhausted or encountered errors.")
        return ""

    def _extract_code(self, raw_text: str) -> str:
        """
        Extracts code enclosed inside <code>...</code> tags, with fallbacks for markdown code blocks.
        """
        # Primary target: strict <code>...</code> XML tags
        match = re.search(r"<code>(.*?)</code>", raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback 1: ```python ... ``` markdown blocks
        markdown_match = re.search(r"```python(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
        if markdown_match:
            return markdown_match.group(1).strip()
            
        # Fallback 2: Generic ``` ... ``` blocks
        generic_match = re.search(r"```(.*?)```", raw_text, re.DOTALL)
        if generic_match:
            return generic_match.group(1).strip()
            
        return raw_text.strip()

    def _build_coder_prompt(self, task: str, plan: str) -> str:
        """
        Constructs a hardened system prompt designed to enforce robust Python coding standards.
        """
        return f"""You are an Expert Autonomous Python Data Science Engineer writing a script for a Kaggle environment.

### OVERARCHING TASK OBJECTIVE:
{task}

### STEP-BY-STEP IMPLEMENTATION PLAN:
{plan}

---

### CRITICAL PYTHON SYNTAX & STRING FORMATTING RULES (STRICT COMPLIANCE):
1. **NO UNTERMINATED STRINGS:** Never insert raw literal newlines inside standard single/double quotes in `print()` statements. Use `\n` or triple-quoted strings (`\"\"\"`).
2. **FORWARD SLASHES ONLY:** Always use forward slashes for Linux paths (e.g., `/kaggle/input/...` or `./output.csv`). NEVER use unescaped backslashes like `\kaggle\input` or `\d+` inside standard strings.
3. **RAW STRINGS FOR REGEX:** Always use raw strings (`r"..."`) when declaring regex patterns or file search expressions.
4. **EXPLICIT ERROR HANDLING:** Wrap file opening, SQLite connections, and model training in `try-except` blocks with informative `print()` statements.
5. **SAVING ARTIFACTS:** Save all artifacts, CSV outputs, and saved models to `./` (current working directory).
6. **MATPLOTLIB & PLOTS:** Set `matplotlib.use('Agg')` BEFORE importing `pyplot`. Save plots via `plt.savefig('filename.png')` and call `plt.close()`. NEVER use `plt.show()`.
7. **CATEGORICAL DATA IN LIGHTGBM:** Explicitly handle categoricals via `df[col] = df[col].astype('category')` or standard label encoding.

### STRICT STRING, F-STRING, AND PRINTING RULES (CRITICAL):
# 1. NO NEWLINE CHARACTERS: You are STRICTLY FORBIDDEN from using the newline character (`\\n`) inside ANY strings, f-strings, or print statements.
# 2. NO MULTILINE STRINGS: Do not span standard single (') or double (") quotes across multiple lines. 
# 3. THE SAFE FORMATTING OPTION: If you need to output multiple lines of text (like a classification report or summary), you MUST use multiple independent `print()` statements.
# 
# ❌ BAD (Will cause SyntaxErrors):
# print("Classification Report:\\n" + report)
# print(f"Metrics:\\nAccuracy: {{acc}}")
# 
# ✅ GOOD (Solid formatting):
# print("Classification Report:")
# print(report)
# print("Metrics:")
# print(f"Accuracy: {{acc}}")
---

### OUTPUT FORMAT INSTRUCTIONS:
* Output ONLY valid, complete, and fully self-contained Python code.
* You MUST enclose the entire code snippet inside <code> and </code> XML tags.
* Do NOT include conversational chatter, markdown wrappers, or intro text outside the <code> tags.

Example Output Structure:
<code>
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

print("🚀 Starting pipeline execution...")
# Code implementation here
</code>

Generate the complete executable Python script now:
"""