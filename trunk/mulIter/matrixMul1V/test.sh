for matrixSize in 1024 2048 3072 4096 5120 6144 7168
do
	 echo "./oclMatrixMul $matrixSize"
	 ./oclMatrixMul $matrixSize >> log.txt
	 ./oclMatrixMul $matrixSize >> log.txt
	 ./oclMatrixMul $matrixSize >> log.txt
done

for matrixSize in 1024 2048 3072 4096 5120 6144 7168
do
	 echo "./virMatrixMul $matrixSize"
	  mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./virMatrixMul $matrixSize >> logv.txt
	  mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./virMatrixMul $matrixSize >> logv.txt
	  mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./virMatrixMul $matrixSize >> logv.txt
done
