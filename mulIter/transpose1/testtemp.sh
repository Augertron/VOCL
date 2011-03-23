#for matrixSize in 1024 2048 3072 4096 5120 6144 7168 
#do
#	 ./tranO --width=$matrixSize --height=$matrixSize >> logo.txt 
#	 ./tranO --width=$matrixSize --height=$matrixSize >> logo.txt
#	 ./tranO --width=$matrixSize --height=$matrixSize >> logo.txt
#done

for matrixSize in 1024 2048 3072 4096 5120 6144 7168 
do
	echo $matrixSize
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./tranV --width=$matrixSize --height=$matrixSize >> logv.txt 
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./tranV --width=$matrixSize --height=$matrixSize >> logv.txt 
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./tranV --width=$matrixSize --height=$matrixSize >> logv.txt 
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./tranV --width=$matrixSize --height=$matrixSize >> logv.txt 
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./tranV --width=$matrixSize --height=$matrixSize >> logv.txt 
done
