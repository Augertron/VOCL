iteration=(40 30 20 20 20 20 20)
bodynum=(15360 23040 30720 38400 46080 53760 61440)
for j in 0 1 2 3 4 5
do
	echo "mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt  -np 1 ./nbodyV --n=${bodynum[j]} --iter=${iteration[j]} >> logv.txt"
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt  -np 1 ./nbodyV --n=${bodynum[j]} --iter=${iteration[j]} >> logv.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt  -np 1 ./nbodyV --n=${bodynum[j]} --iter=${iteration[j]} >> logv.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt  -np 1 ./nbodyV --n=${bodynum[j]} --iter=${iteration[j]} >> logv.txt
done


