#!/bin/bash

sbatchfile="build_dataset_run.sbatch"
LIST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
k=1
for x in ${LIST[@]}; do
	OLDTXT=`grep "build_dataset.py" $sbatchfile`
	sed -i "s|$OLDTXT|srun python /scratch/users/wdu/code/build_dataset.py --test 0 --train_num $x|" $sbatchfile
	OLDJOB=`grep "job-name=" $sbatchfile`
	sed -i "s|$OLDJOB|\#SBATCH --job-name=wfdg$k|" $sbatchfile
	OLDOUT=`grep "output=" $sbatchfile`
	sed -i "s|$OLDOUT|\#SBATCH --output=wfdg$k.out|" $sbatchfile
	OLDERR=`grep "error=" $sbatchfile`
	sed -i "s|$OLDERR|\#SBATCH --error=wfdg$k.err|" $sbatchfile
	sbatch $sbatchfile
  k=$((k+1))
done
