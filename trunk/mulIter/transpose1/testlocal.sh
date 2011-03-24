#iteration=(400 200 150 100 100 100)
iteration=(200 100 100 80 80 80)
matrixSize=(1024 2048 3072 4096 5120 6144 7168)
for j in 0 1 2 3 4 5
do
	 echo "mpiexec.hydra -np 1 -binding user:0 ./tranO --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt"
	 mpiexec.hydra -np 1 -binding user:0 ./tranO --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt 
	 mpiexec.hydra -np 1 -binding user:0 ./tranO --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt 
	 mpiexec.hydra -np 1 -binding user:0 ./tranO --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt 
done

for j in 0 1 2 3 4 5
do
	 echo "mpiexec.hydra -np 1 -binding user:8 ./tranO --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt"
	 mpiexec.hydra -np 1 -binding user:8 ./tranO --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt 
	 mpiexec.hydra -np 1 -binding user:8 ./tranO --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt 
	 mpiexec.hydra -np 1 -binding user:8 ./tranO --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt 
done

