#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL

lscpu
python new_full_setup.py
