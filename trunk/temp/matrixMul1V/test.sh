for matrixSize in 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240
do
	 echo "./oclMatrixMul $matrixSize"
	 ./oclMatrixMul $matrixSize
	 ./oclMatrixMul $matrixSize
	 ./oclMatrixMul $matrixSize
done

for matrixSize in 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240
do
	 echo "./virMatrixMul $matrixSize"
	 mpiexec -hosts gpu0031,gpu0032 -np 1 ./virMatrixMul $matrixSize
	 mpiexec -hosts gpu0031,gpu0032 -np 1 ./virMatrixMul $matrixSize
	 mpiexec -hosts gpu0031,gpu0032 -np 1 ./virMatrixMul $matrixSize
done
