iteration=(100 100 40 40 20 20 20)
matrixSize=(1024 2048 3072 4096 5120 6144 7168)

for j in 0 1 2 3 4 5
do
	echo "mpiexec.hydra -np 1 -binding user:0 ./oclMatrixMul ${matrixSize[j]} ${iteration[j]} >> log.txt"
	mpiexec.hydra -np 1 -binding user:0 ./oclMatrixMul ${matrixSize[j]} ${iteration[j]} >> log.txt
	mpiexec.hydra -np 1 -binding user:0 ./oclMatrixMul ${matrixSize[j]} ${iteration[j]} >> log.txt
	mpiexec.hydra -np 1 -binding user:0 ./oclMatrixMul ${matrixSize[j]} ${iteration[j]} >> log.txt
done

for j in 0 1 2 3 4 5
do
	echo "mpiexec.hydra -np 1 -binding user:8 ./oclMatrixMul ${matrixSize[j]} ${iteration[j]} >> log.txt"
	mpiexec.hydra -np 1 -binding user:8 ./oclMatrixMul ${matrixSize[j]} ${iteration[j]} >> log.txt
	mpiexec.hydra -np 1 -binding user:8 ./oclMatrixMul ${matrixSize[j]} ${iteration[j]} >> log.txt
	mpiexec.hydra -np 1 -binding user:8 ./oclMatrixMul ${matrixSize[j]} ${iteration[j]} >> log.txt
done


