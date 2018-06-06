#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name
#SBATCH --job-name=wfd0
#################  
#a file for job output, you can check job progress
#SBATCH --output=wfd0.out
#################
# a file for errors from the job
#SBATCH --error=wfd0.err
#################
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=2:00:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=16000
#you could use --mem-per-cpu; they mean what we are calling cores
#################

module load python/3.4.3

srun python /scratch/users/wdu/code/build_dataset.py --test 1
