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

    def assign_queues(self, task: dict) -> tuple[list[str], list[str], list[str]]:
        """
        Returns a tuple of 3 model queues: (planner_queue, coder_queue, log_queue)
        """
        default_models = ['gemini-3.5-flash-lite', 'gemini-3.6-flash','gemini-3.5-flash-lite']
        requested_models = task.get('models') or default_models
        
        queues = [self._build_queue(m) for m in requested_models]

        # Ensure we always have at least 3 elements to allow 3-variable unpacking
        while len(queues) < 3:
            queues.append(queues[-1]) # Reuse the last queue for cognitive/log step

        return queues[0], queues[1], queues[2]