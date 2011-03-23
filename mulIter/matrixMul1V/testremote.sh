iteration=(100 100 40 40 20 20 20)
matrixSize=(1024 2048 3072 4096 5120 6144 7168)

for j in 0 1 2 3 4 5
do
	echo "mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./virMatrixMul ${matrixSize[j]} ${iteration[j]} >> log.txt"
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./virMatrixMul ${matrixSize[j]} ${iteration[j]} >> log.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./virMatrixMul ${matrixSize[j]} ${iteration[j]} >> log.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./virMatrixMul ${matrixSize[j]} ${iteration[j]} >> log.txt
done


