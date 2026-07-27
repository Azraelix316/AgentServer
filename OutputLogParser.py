import os

class OutputLogParser:
    def __init__(self, max_head_lines: int = 50):
        """
        Initializes the parser.
        :param max_head_lines: Maximum number of lines to read from the top of non-log output files.
        """
        self.max_head_lines = max_head_lines

    def parse_directory(self, local_output_dir: str) -> dict:
        """
        Sweeps local_output_dir to:
        1. Read the first `max_head_lines` of non-.log files (CSVs, TXT, JSON, etc.).
        2. Filter .log files specifically for error indicators (stderr, error, traceback).
        3. Safely handle non-UTF-8 / binary files (images, model weights, checkpoints).

        Returns:
            dict: {"heads": str, "stderr": str}
        """
        if not os.path.exists(local_output_dir):
            return {
                "heads": f"Error: Directory '{local_output_dir}' does not exist.",
                "stderr": f"Error: Directory '{local_output_dir}' does not exist."
            }

        non_log_heads = []
        stderr_logs = []

        print(f"📄 Parsing output directory '{local_output_dir}'...")

        for root, _, files in os.walk(local_output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, local_output_dir)

                if file.endswith('.log'):
                    # Filter .log files strictly for error indicators
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            filtered_lines = [
                                line for line in f 
                                if any(err_term in line.lower() for err_term in ['stderr', 'error', 'exception', 'traceback'])
                            ]
                            if filtered_lines:
                                stderr_logs.append(f"--- {rel_path} (stderr/error lines) ---\n" + "".join(filtered_lines))
                    except Exception as e:
                        stderr_logs.append(f"⚠️ Error reading log {rel_path}: {e}\n")

                else:
                    # Read heads of standard data/output files
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            head_lines = []
                            for i, line in enumerate(f):
                                if i >= self.max_head_lines:
                                    head_lines.append(f"\n... (truncated after {self.max_head_lines} lines)\n")
                                    break
                                head_lines.append(line)
                            
                            if head_lines:
                                non_log_heads.append(f"--- {rel_path} (head) ---\n" + "".join(head_lines) + "\n")

                    except UnicodeDecodeError:
                        # Gracefully handle binary artifacts (.png, .h5, .parquet, .pkl, .zip)
                        non_log_heads.append(f"--- {rel_path} ---\n(Binary artifact created - Non-UTF-8 content)\n\n")
                    except Exception as e:
                        non_log_heads.append(f"⚠️ Error reading {rel_path}: {e}\n\n")

        return {
            "heads": "".join(non_log_heads) if non_log_heads else "No standard output files generated.",
            "stderr": "".join(stderr_logs) if stderr_logs else "No stderr or error logs detected."
        }