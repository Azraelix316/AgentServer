import google.generativeai as genai

class PlannerAgent:
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-3.5-flash-lite"):
        """
        Initializes the Planner Agent, responsible for strategizing the next coding step.
        """
        genai.configure(api_key=gemini_api_key, transport="rest")
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
Note: Heavy model weights (.safetensors, .pth, .bin) and database files generated during execution are retained in Kaggle storage and automatically excluded from local log syncs to save disk space.
[Execution Errors (STDERR)]:
{last_stderr if last_stderr else "No errors detected in the last run."}

### INSTRUCTIONS FOR YOUR PLAN:
1. Assess the results of the last execution based on the Errors and Output Heads. If there were errors, your plan MUST prioritize fixing them.
2. Formulate the immediate next steps required to progress toward the OVERARCHING GOAL.
3. Be explicitly clear about what data transformations, machine learning models, or file saving steps the Coder needs to write.
4. Keep the plan focused on the *very next* script to be written. Do not plan 10 steps ahead if step 1 is currently failing.
5. Format your output as a clean, actionable Markdown checklist or numbered list. Do not write any code yourself.
6. If the task is fully accomplished, instead print out "TASK_COMPLETE" which will stop all services.

### CRITICAL RULE: DECLARING TASK COMPLETION
- **NEVER** output `TASK_COMPLETE` on the first iteration or while simply generating a plan. Creating a plan is NOT completing the task.
- You may ONLY output `TASK_COMPLETE` if ALL of the following criteria are met:
  1. At least one code execution step has successfully run.
  2. The actual code execution output explicitly verifies that the desired end-result (e.g., submission file, trained model artifact, or verified metrics) was produced and saved.
  3. No pending steps remain in the action plan.
Write the plan now:
"""
    def plan_from_forked(
        self,
        new_task_prompt: str,
        status_content: str,
        latest_action_code: str,
        report_content: str
    ) -> str:
        """
        Generates an initial execution plan for a NEW task forked from a parent task.
        Uses parent task artifacts (status.txt, latest_action.txt, report.txt) as context/boosters.
        """
        print("🔀 Planner Agent is bootstrapping plan for a FORKED task using parent context...")

        prompt = self._build_forked_planner_prompt(
            new_task_prompt,
            status_content,
            latest_action_code,
            report_content
        )

        try:
            response = self.model.generate_content(prompt)
            print("✅ Plan for forked task generated successfully.")
            return response.text
        except Exception as e:
            error_msg = f"❌ Gemini API Error in Planner (Forked Plan): {str(e)}"
            print(error_msg)
            return error_msg

    def _build_forked_planner_prompt(
        self,
        new_task_prompt: str,
        status_content: str,
        latest_action_code: str,
        report_content: str
    ) -> str:
        """
        Constructs the instructional prompt for initializing a forked task.
        """
        return f"""You are the Planner Module of an autonomous AI coding agent.
This is Iteration 1 of a NEW TASK that has been forked from a prior task. 
YOU ARE NOT DONE YET! YOU ARE MERELY GETTING THE REPORTS OF A PAST VERSION! DO NOT OUTPUT TASK COMPLETE!
================================================================================
SECTION 1: HISTORICAL PARENT CONTEXT (READ-ONLY REFERENCE / BOOSTER)
================================================================================
Use the historical artifacts below to understand what methods, feature processing steps, 
or code structures worked in the parent task. DO NOT attempt to finish or continue 
the parent task's original goal. Treat this purely as domain reference.

[Parent Task Execution Status (status.txt)]:
{status_content if status_content else "No status available."}

[Parent Task Last Ran Code (latest_action.txt)]:
```python
{latest_action_code if latest_action_code else "# No code recorded from parent task."}
[Parent Task Analytical Report (report.txt)]:
{report_content if report_content else "No analytical report available."}

================================================================================
SECTION 2: YOUR ACTIVE NEW TASK CONTRACT (PRIMARY OBJECTIVE)
NEW OVERARCHING GOAL:
{new_task_prompt}

INSTRUCTIONS FOR YOUR PLAN:
Carefully review the NEW OVERARCHING GOAL above.

Identify code snippets, feature pipelines, or data preparation logic from latest_action.txt or report.txt that can be reused to kickstart this new task.

Formulate a step-by-step, actionable Markdown checklist for the Coder Module to build the initial Python script for this new task.

Clearly specify what code to adapt from the parent's latest_action.txt versus what new logic needs to be written.

Do not write any code yourself; produce a structured Markdown plan.

Reset all assumptions. This is Iteration 1 of the new goal.

Write the initial plan for the new task now:
"""