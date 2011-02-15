#for matrixSize in 1024 2048 3072 4096 5120 6144 7168
#do
#	 ./oclMatrixMul $matrixSize
#	 ./oclMatrixMul $matrixSize
#	 ./oclMatrixMul $matrixSize
#done

for matrixSize in 1024 2048 3072 4096 5120 6144 7168
do
	 mpiexec -hosts bb30,bb31 -np 1 ./virMatrixMul $matrixSize
	 mpiexec -hosts bb30,bb31 -np 1 ./virMatrixMul $matrixSize
	 mpiexec -hosts bb30,bb31 -np 1 ./virMatrixMul $matrixSize
done	
