#!/bin/bash
#SBATCH --job-name=test_python_simple
#SBATCH --output=%x_%j.out
#SBATCH --mail-user=j.a.krzywda@liacs.leidenuniv.nl
#SBATCH --mail-type="ALL"
#SBATCH --mem=100M
#SBATCH --time=00:05:00
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change

module load Python/3.9.5-GCCcore-10.3.0

# Source the Python virtual environment
export CWD=$(pwd)

source $CWD/python_test_venv/bin/activate
echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"

# get the current working directory

echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python3 test.py
echo "[$SHELL] ## Script finished"
echo "[$SHELL] #### Finished Python test. Have a nice day"