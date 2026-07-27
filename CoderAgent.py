import re
import google.generativeai as genai

class CoderAgent:
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-3.6-flash"):
        """
        Initializes the Coder Agent, responsible for turning plans into Kaggle-ready Python code.
        """
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_code(self, original_task: str, current_plan: str) -> str:
        """
        Takes the planner's checklist and generates the corresponding Python script.
        Extracts the code cleanly from the <code> XML tags.
        """
        print("💻 Coder Agent is writing the Python script based on the current plan...")
        
        prompt = self._build_coder_prompt(original_task, current_plan)

        try:
            response = self.model.generate_content(prompt)
            raw_output = response.text
            
            # Extract the code using Regex to target the XML blocks
            clean_code = self._extract_code(raw_output)
            
            if not clean_code:
                raise ValueError("LLM failed to wrap code in <code> tags.")
                
            print("✅ Code generated and parsed successfully.")
            return clean_code
            
        except Exception as e:
            error_msg = f"❌ Gemini API Error in Coder: {str(e)}"
            print(error_msg)
            return ""

    def _extract_code(self, raw_text: str) -> str:
        """
        Uses regex to find the content specifically inside <code> </code> tags.
        Handles cases where the LLM might add extra chatter before or after the tags.
        """
        match = re.search(r"<code>(.*?)</code>", raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            # Strip leading/trailing whitespace but maintain internal indentation
            return match.group(1).strip()
        
        # Fallback: Check if they used markdown python blocks instead despite instructions
        markdown_match = re.search(r"```python(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
        if markdown_match:
            return markdown_match.group(1).strip()
            
        return ""

    def _build_coder_prompt(self, task: str, plan: str) -> str:
        """
        Constructs the strict system prompt for the coding agent.
        """
        return f"""You are the Coder Module of an autonomous AI data science agent.
Your objective is to write Python 3 code that will execute in a headless Kaggle Docker environment.

### OVERARCHING GOAL:
{task}

### YOUR SPECIFIC PLAN FOR THIS SCRIPT:
{plan}

### KAGGLE ENVIRONMENT RULES (CRITICAL):
1. **Inputs:** If loading external data, assume standard data science libraries are available (pandas, numpy, scikit-learn, etc.).
2. **Outputs:** You MUST save all artifacts, plots, CSVs, or models to the current working directory: `./` (which maps to `/kaggle/working/` in the environment). Do not use absolute paths.
3. **No Interactive Displays:** Do not use `plt.show()`. Save plots using `plt.savefig('my_plot.png')`.
4. **Logging:** Print meaningful status updates and metric results to stdout so the evaluation module can read them.

### OUTPUT FORMAT REQUIREMENTS:
You must output ONLY valid Python code. 
You MUST wrap your entire code block inside strict XML tags: <code> and </code>.
Do not write any introductory or concluding text outside of these tags. 

Example:
<code>
import pandas as pd
print("Starting execution...")
# ... rest of code
</code>

Write the Python code now:
"""