#!/bin/bash
#SBATCH --job-name=test_python_simple
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="<your_email_address>"
#SBATCH --mail-type="ALL"
#SBATCH --mem=100M
#SBATCH --time=00:05:00
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


# Source the Python virtual environment

source /data1/krzywdaja/test_env
echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD
# Run the file
echo "[$SHELL] ## Run script"
python3 test.py
echo "[$SHELL] ## Script finished"
echo "[$SHELL] #### Finished Python test. Have a nice day"