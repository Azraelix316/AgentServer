import google.generativeai as genai

class PlannerAgent:
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-3.5-flash-lite"):
        """
        Initializes the Planner Agent, responsible for strategizing the next coding step.
        """
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_plan(
        self, 
        original_task_prompt: str, 
        memory_content: str, 
        last_heads: str, 
        last_stderr: str
    ) -> str:
        """
        Reads the memory and the immediate results of the last run, 
        then generates a clear plan for the Coder agent.
        """
        print("🧭 Planner Agent is analyzing the current state and drafting the next plan...")
        
        prompt = self._build_planner_prompt(
            original_task_prompt, 
            memory_content, 
            last_heads, 
            last_stderr
        )

        try:
            # We use standard text generation here, as the coder usually 
            # does better reading a structured markdown plan rather than raw JSON.
            response = self.model.generate_content(prompt)
            print("✅ Plan generated successfully.")
            return response.text
        except Exception as e:
            error_msg = f"❌ Gemini API Error in Planner: {str(e)}"
            print(error_msg)
            return error_msg

    def _build_planner_prompt(
        self, 
        task_prompt: str, 
        memory_content: str, 
        last_heads: str, 
        last_stderr: str
    ) -> str:
        """
        Constructs the strict instructional prompt for the Planner.
        """
        return f"""You are the Planner Module of an autonomous AI coding agent.
Your job is to write a highly specific, step-by-step plan for the Coder Module. The Coder will use your plan to write a Python script for execution in a Kaggle environment.

### OVERARCHING GOAL:
{task_prompt}

### AGENT MEMORY (Context of past actions):
{memory_content if memory_content else "No prior memory. This is the first step."}

### RESULTS FROM THE LAST EXECUTION:
[Output File Heads]:
{last_heads if last_heads else "No outputs generated in the last run."}

[Execution Errors (STDERR)]:
{last_stderr if last_stderr else "No errors detected in the last run."}

### INSTRUCTIONS FOR YOUR PLAN:
1. Assess the results of the last execution based on the Errors and Output Heads. If there were errors, your plan MUST prioritize fixing them.
2. Formulate the immediate next steps required to progress toward the OVERARCHING GOAL.
3. Be explicitly clear about what data transformations, machine learning models, or file saving steps the Coder needs to write.
4. Keep the plan focused on the *very next* script to be written. Do not plan 10 steps ahead if step 1 is currently failing.
5. Format your output as a clean, actionable Markdown checklist or numbered list. Do not write any code yourself.

Write the plan now:
"""