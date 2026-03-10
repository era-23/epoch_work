# Gemini-assisted

import argparse
import json
import os
from pathlib import Path

def generate_slurm_content(config):
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
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        job_list = json.load(f)

    for job_config in job_list:
        out_file = job_config.get("filename", f"{job_config.get('job_name', 'job')}.sh")
        content = generate_slurm_content(job_config)
        
        with open(out_file, "w") as f:
            f.write(content)
        
        print(f"Generated: {out_file} (Time Limit: {job_config.get('time', '02:00:00')})")