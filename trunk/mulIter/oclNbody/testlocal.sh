iteration=(40 30 20 20 20 20 20)
bodynum=(15360 23040 30720 38400 46080 53760 61440)

for j in 0 1 2 3 4 5
do
	echo "mpiexec.hydra -np 1 -binding user:0 ./nbodyO --n=${bodynum[j]} --iter=${iteration[j]}>> logo.txt"
	mpiexec.hydra -np 1 -binding user:0 ./nbodyO --n=${bodynum[j]} --iter=${iteration[j]}>> logo.txt
	mpiexec.hydra -np 1 -binding user:0 ./nbodyO --n=${bodynum[j]} --iter=${iteration[j]}>> logo.txt
	mpiexec.hydra -np 1 -binding user:0 ./nbodyO --n=${bodynum[j]} --iter=${iteration[j]}>> logo.txt
done

for j in 0 1 2 3 4 5
do
	echo "mpiexec.hydra -np 1 -binding user:8 ./nbodyO --n=${bodynum[j]} --iter=${iteration[j]}>> logo.txt"
	mpiexec.hydra -np 1 -binding user:8 ./nbodyO --n=${bodynum[j]} --iter=${iteration[j]}>> logo.txt
	mpiexec.hydra -np 1 -binding user:8 ./nbodyO --n=${bodynum[j]} --iter=${iteration[j]}>> logo.txt
	mpiexec.hydra -np 1 -binding user:8 ./nbodyO --n=${bodynum[j]} --iter=${iteration[j]}>> logo.txt
done


