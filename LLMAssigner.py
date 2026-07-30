# LLMAssigner.py

# Global fallback chain ordered from highest capability to lowest
GLOBAL_MODEL_FALLBACK = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-flash",
    "gemini-3.1-flash-lite",
    "gemma-4-31b-it",
    "gemini-2.5-flash",
    "gemma-4-26b-a4b-it"
]

class LLMAssigner:
    def __init__(self, fallback_chain: list[str] = None):
        self.fallback_chain = fallback_chain or GLOBAL_MODEL_FALLBACK

    def _build_queue_for_model(self, requested_model: str) -> list[str]:
        """
        Creates an ordered list starting from requested_model down the global fallback chain.
        """
        if requested_model in self.fallback_chain:
            start_idx = self.fallback_chain.index(requested_model)
            return self.fallback_chain[start_idx:]
        
        # If model isn't in global chain, put it first, then append full chain
        return [requested_model] + self.fallback_chain

    def assign_queues(self, task: dict) -> list[list[str]]:
        """
        Takes a task dict, extracts requested models (defaulting if missing),
        and returns a list of model_queues (e.g. [[planner_queue], [coder_queue]])
        """
        # Default requested models if none provided
        is_private = task.get('is_private', False)
        default_models = ['gemini-3.5-flash-lite', 'gemini-3.6-flash','gemini-3.5-flash-lite'] if is_private else ['gemini-3.5-flash-lite', 'gemini-3.5-flash-lite','gemini-3.5-flash-lite']
        
        requested_models = task.get('models') or default_models

        # Build fallback array for each agent/step requested
        model_queues = [self._build_queue_for_model(m) for m in requested_models]
        return model_queues