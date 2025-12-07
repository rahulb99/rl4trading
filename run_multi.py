import json
import re
import subprocess
import sys
import os

# ---------------- CONFIGURATION ---------------- #

# List the exact "name" of the configs you want to run from launch.json
# You can also use partial strings if you enable exact_match=False below.
TARGETS = [
    # "Mag7 DDPG",
    # "Mag7 MultiDiscrete A2C",
    # "Mag7 MultiDiscrete REINFORCE",
    # "AAPL MultiDiscrete REINFORCE",
    # "AAPL MultiDiscrete A2C",
    # "AAPL DDPG",
    # "AAPL Discrete DQN",
    # "AAPL Discrete DDQN",
]

# Set to False to run any config that CONTAINS the string (e.g. "Mag7" runs all Mag7)
EXACT_MATCH = True 

# Set to True to run in parallel, False to run one after another
RUN_PARALLEL = False
RUN_PARALLEL_CNT = 1

# Path to your launch.json
LAUNCH_PATH = ".vscode/launch.json"

# ----------------------------------------------- #

def remove_comments(json_str):
    """Removes C-style // comments and /* */ block comments."""
    pattern = r"//.*?$|/\*.*?\*/"
    return re.sub(pattern, "", json_str, flags=re.MULTILINE | re.DOTALL)

def remove_trailing_commas(json_str):
    """Removes trailing commas which are valid in VSCode but invalid in standard JSON."""
    # Remove comma before close curly brace }
    json_str = re.sub(r',(\s*})', r'\1', json_str)
    # Remove comma before close square bracket ]
    json_str = re.sub(r',(\s*])', r'\1', json_str)
    return json_str

def get_launch_configs():
    if not os.path.exists(LAUNCH_PATH):
        print(f"Error: Could not find {LAUNCH_PATH}")
        sys.exit(1)

    with open(LAUNCH_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Strip Comments
    content = remove_comments(content)
    # 2. Strip Trailing Commas (The fix for your error)
    content = remove_trailing_commas(content)
    
    try:
        data = json.loads(content)
        return data.get("configurations", [])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        # Print a snippet of where it failed to help debug
        lines = content.splitlines()
        if e.lineno <= len(lines):
            print(f"Problem likely near line {e.lineno}:")
            print(lines[e.lineno-1].strip())
        sys.exit(1)

def run():
    configs = get_launch_configs()
    processes = []

    print(f"--- Scanning launch.json for targets: {TARGETS} ---")

    count = 0
    for config in configs:
        cfg_name = config.get("name", "")
        
        # Check if this config matches our targets
        should_run = False
        if EXACT_MATCH:
            if cfg_name in TARGETS:
                should_run = True
        else:
            if any(t in cfg_name for t in TARGETS):
                should_run = True

        if should_run:
            count += 1
            program = config.get("program")
            args = config.get("args", [])

            # Handle ${file} or special VS Code variables
            if "${" in program or any("${" in str(a) for a in args):
                print(f"Skipping '{cfg_name}': Contains VS Code variables like ${{file}}")
                continue

            # Construct command
            cmd = [sys.executable, program] + args
            
            print(f"[{'PARALLEL' if RUN_PARALLEL else 'SEQUENTIAL'}] Launching: {cfg_name}")
            
            if RUN_PARALLEL:
                p = subprocess.Popen(cmd)
                processes.append((cfg_name, p))
            else:
                subprocess.run(cmd)

    if count == 0:
        print("No matching configurations found. Check your TARGETS list.")

    if RUN_PARALLEL and processes:
        print(f"\n--- {len(processes)} tasks running in background. Waiting... ---")
        try:
            for name, p in processes:
                p.wait()
                print(f"Finished: {name} (Exit Code: {p.returncode})")
        except KeyboardInterrupt:
            print("\nForce stopping all processes...")
            for _, p in processes:
                p.terminate()

if __name__ == "__main__":
    run()