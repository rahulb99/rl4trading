import json
import re
import subprocess
import sys
import os
import time
import argparse

# ---------------- CONFIGURATION ---------------- #

# Master list of available configurations (corresponds to "name" in launch.json)
TARGETS = [
    "Mag7 DDPG",                    # Index 0
    "Mag7 MultiDiscrete A2C",       # Index 1
    "Mag7 MultiDiscrete REINFORCE", # Index 2
    "Mag7 Discrete DQN",            # Index 3
    "Mag7 Discrete DDQN",           # Index 4
]

# Path to your launch.json
LAUNCH_PATH = ".vscode/launch.json"

# ----------------------------------------------- #

def remove_comments(json_str):
    """Removes C-style // comments and /* */ block comments."""
    pattern = r"//.*?$|/\*.*?\*/"
    return re.sub(pattern, "", json_str, flags=re.MULTILINE | re.DOTALL)

def remove_trailing_commas(json_str):
    """Removes trailing commas which are valid in VSCode but invalid in standard JSON."""
    json_str = re.sub(r',(\s*})', r'\1', json_str)
    json_str = re.sub(r',(\s*])', r'\1', json_str)
    return json_str

def get_launch_configs():
    if not os.path.exists(LAUNCH_PATH):
        print(f"Error: Could not find {LAUNCH_PATH}")
        sys.exit(1)

    with open(LAUNCH_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    content = remove_comments(content)
    content = remove_trailing_commas(content)
    
    try:
        data = json.loads(content)
        return data.get("configurations", [])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        lines = content.splitlines()
        if e.lineno <= len(lines):
            print(f"Problem likely near line {e.lineno}:")
            print(lines[e.lineno-1].strip())
        sys.exit(1)

def print_targets():
    print("\nAvailable Configurations:")
    print("-" * 30)
    for i, target in enumerate(TARGETS):
        print(f"[{i}] {target}")
    print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description="Run VS Code launch configurations by index.")
    
    # Argument: indices (list of integers)
    parser.add_argument(
        "indices", 
        metavar="N", 
        type=int, 
        nargs="*", 
        help="Space-separated list of indices of launch configurations to run (e.g., 0 2 5)"
    )

    # Argument: Parallel Flag
    parser.add_argument(
        "-p", "--parallel", 
        action="store_true", 
        help="Run the selected configurations in parallel."
    )

    # Argument: List Flag
    parser.add_argument(
        "-l", "--list", 
        action="store_true", 
        help="List all available configuration targets and their indices."
    )

    args = parser.parse_args()

    # 1. Handle --list
    if args.list:
        print_targets()
        return

    # 2. Validate Input
    if not args.indices:
        print("No indices provided.")
        print_targets()
        print("Usage example: python run_multi.py 0 5 --parallel")
        return

    # 3. Validate Indices
    selected_names = []
    print("--- Selection ---")
    for idx in args.indices:
        if 0 <= idx < len(TARGETS):
            name = TARGETS[idx]
            print(f"✅ Index {idx}: {name}")
            selected_names.append(name)
        else:
            print(f"❌ Index {idx} is out of range.")
    
    if not selected_names:
        print("No valid configurations selected. Exiting.")
        return

    # 4. Find Configs in launch.json
    all_configs = get_launch_configs()
    cmds_to_run = []

    for name in selected_names:
        target_config = next((c for c in all_configs if c.get("name") == name), None)
        
        if not target_config:
            print(f"⚠️ Warning: Could not find '{name}' in launch.json. Skipping.")
            continue

        program = target_config.get("program")
        args_list = target_config.get("args", [])
        cwd = target_config.get("cwd", os.getcwd())

        # Check for VS Code vars
        if "${" in program or any("${" in str(a) for a in args_list):
            print(f"⚠️ Skipping '{name}': Contains variables like ${{file}}.")
            continue

        cmd = [sys.executable, program] + args_list
        cmds_to_run.append((name, cmd, cwd))

    if not cmds_to_run:
        print("Nothing to run.")
        return

    # 5. Execute
    processes = []
    mode = "PARALLEL" if args.parallel else "SEQUENTIAL"
    print(f"\n--- Launching {len(cmds_to_run)} Tasks ({mode}) ---")

    try:
        for name, cmd, cwd in cmds_to_run:
            print(f"🚀 Launching: {name}")
            
            if args.parallel:
                p = subprocess.Popen(cmd, cwd=cwd)
                processes.append((name, p))
                time.sleep(0.2) # Stagger slightly
            else:
                subprocess.run(cmd, cwd=cwd)
                print(f"Finished: {name}")

        # 6. Wait if Parallel
        if args.parallel and processes:
            print(f"\n--- Waiting for {len(processes)} processes to finish... ---")
            for name, p in processes:
                p.wait()
                print(f"✅ Completed: {name} (Exit Code: {p.returncode})")

    except KeyboardInterrupt:
        print("\n🛑 Keyboard Interrupt. Stopping all processes...")
        for _, p in processes:
            p.terminate()

if __name__ == "__main__":
    main()