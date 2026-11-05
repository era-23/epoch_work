# Gemini-assisted

import argparse
import json
import os
from pathlib import Path

def generate_slurm_cirrus(config):
    """Formats the Slurm script string using dictionary values for Cirrus."""
    # Configuration extraction with defaults
    job_name = config.get("job_name", "slurm_job")
    mem = config.get("mem", None)  # Cirrus allocates memory by node/core by default
    cpus = config.get("cpus", 1)
    ntasks = config.get("ntasks", 1)
    time_limit = config.get("time", "02:00:00")
    script = config.get("script", "main.py")
    log_dir = config.get("output", ".")
    account = config.get("account", "e281-ypimcf")  # Set your Cirrus budget code

    # GPU vs CPU partitioning for Cirrus
    gpu_request = config.get("gpu")
    if gpu_request:
        partition = config.get("partition", "gpu")
        qos = config.get("qos", "gpu")
        gpu_line = f"#SBATCH --gpus={gpu_request}\n"
    else:
        partition = config.get("partition", "standard")
        qos = config.get("qos", "standard")
        gpu_line = ""

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, f"{job_name}_%j.log")

    # Optional Memory flag (only add if explicitly specified)
    mem_line = f"#SBATCH --mem={mem}\n" if mem else ""

    header = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus}
{mem_line}{gpu_line}#SBATCH --time={time_limit}
#SBATCH --account={account}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=era536@york.ac.uk
#SBATCH --output={log_path}
"""

    script_args_list = config.get("script_args", [])
    if isinstance(script_args_list, list):
        # Format as array elements: "  --arg1" \n "  --arg2"
        array_items = "\n    ".join([f'"{arg}"' for arg in script_args_list])
        args_block = f"ARGS=(\n    {array_items}\n)"
    else:
        args_block = f'ARGS="{script_args_list}"'

    return header + f"""
# Define long argument list as a Bash array
{args_block}

module restore
source .venv/bin/activate

python {script} "${{ARGS[@]}}"

echo "Job finished at $(date)"
"""

def generate_slurm_viking(config):
    """Formats the Slurm script string using dictionary values."""
    # Configuration extraction with defaults
    job_name = config.get("job_name", "slurm_job")
    mem = config.get("mem", "8G")
    cpus = config.get("cpus", 1)
    ntasks = config.get("ntasks", 1)
    # Default time to 2 hours if not specified
    time_limit = config.get("time", "02:00:00") 
    script = config.get("script", "main.py")
    script_args = str.join(" ", config.get("script_args", ""))
    log_dir = config.get("output", ".")

    # GPU Logic: Add --gres if 'gpu' key is present
    gpu_request = config.get("gpu")
    gpu_line = f"#SBATCH --gres=gpu:{gpu_request}" if gpu_request else ""

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, f"{job_name}_%j.log")

    header = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition=nodes
#SBATCH --mem={mem}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=era536@york.ac.uk
#SBATCH --time={time_limit}
#SBATCH --account=pet-icepic-2024
#SBATCH --output={log_path}
"""

    if gpu_line:
        header += gpu_line

    return header + f"""

echo "Job: {job_name} started at $(date)"
echo "Requested Time: {time_limit}"

# Abort if any command fails
set -e

# purge any existing modules
module purge

# Load modules
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

python {script} {script_args}

echo "Job finished at $(date)"
        """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiple Slurm scripts from a JSON config.")
    parser.add_argument(
        "--config", 
        type=Path, 
        required=True, 
        help="Path to the JSON config file."
    )
    parser.add_argument(
        "--cluster", 
        type=str, 
        required=True, 
        help="Cluster for generting the script. Currently 'viking' or 'cirrus'."
    )
    
    args = parser.parse_args()

    if args.cluster not in ["viking", "cirrus"]:
        print("Unrecognised cluster. Must be one of 'viking' or 'cirrus'")

    with open(args.config, 'r') as f:
        job_list = json.load(f)

    for job_config in job_list:
        out_file = job_config.get("filename", f"{job_config.get('job_name', 'job')}.sh")
        if args.cluster == "viking":
            content = generate_slurm_viking(job_config)
        if args.cluster == "cirrus":
            content = generate_slurm_cirrus(job_config)
        
        with open(out_file, "w") as f:
            f.write(content)
        
        print(f"Generated: {out_file} (Time Limit: {job_config.get('time', '02:00:00')})")