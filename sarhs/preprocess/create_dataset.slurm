#!/bin/bash

##SBATCH --job-name=opt
##SBATCH --partition=sadow
#SBATCH --partition=kill-shared
##SBATCH --account=sadow

## 3 day max run time for public partitions, except 4 hour limit  in sandbox
#SBATCH --time=1-00:00:00 ## time format is DD-HH:MM:SS
## task-per-node x cpus-per-task should not typically exceed core count on an individual node
#SBATCH --nodes=1
##SBATCH --nodelist=gpu-0008
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96000 ## max amount of memory per node you require
##SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node    

##SBATCH --error=hello-%A_%a.err ## %A - filled with jobid
##SBATCH --output=hello-%A_%a.out ## %A - filled with jobid
##SBATCH --error=job-%A.out ## %A - filled with jobid
##SBATCH --output=job-%A.out ## %A - filled with jobid
#SBATCH --error=job.out
#SBATCH --output=job.out

## Use this for email notifications.
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=peter.sadowski@hawaii.edu

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html
#source ~/.bash_profile
#./hello ${SLURM_ARRAY_TASK_ID}
source ~/profiles/auto.profile

jupyter nbconvert create_dataset.ipynb --to python

# It's useful to debug in ipynb, then run train script.
# But the ipynb needs to be able to run as a script.
echo $HOSTNAME
pwd
echo $SHELL
python create_dataset.py
