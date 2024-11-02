#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL

git checkout rafael
rm -r ~/.cache
rm -r __pycache__
python new_full_setup.py
git add .
git commit -m "Finish lab5 job"
git push
