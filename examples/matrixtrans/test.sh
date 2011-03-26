for matrixSize in 1024 2048 3072 4096 5120 6144 7168 
do
	 ./tranO --width=$matrixSize --height=$matrixSize >> logo.txt 
	 ./tranO --width=$matrixSize --height=$matrixSize >> logo.txt
	 ./tranO --width=$matrixSize --height=$matrixSize >> logo.txt
done

for matrixSize in 1024 2048 3072 4096 5120 6144 7168
do
	echo $matrixSize
	mpiexec -hosts gpu0031,gpu0032 -np 1 ./tranV --width=$matrixSize --height=$matrixSize >> logv.txt 
	mpiexec -hosts gpu0031,gpu0032 -np 1 ./tranV --width=$matrixSize --height=$matrixSize >> logv.txt
	mpiexec -hosts gpu0031,gpu0032 -np 1 ./tranV --width=$matrixSize --height=$matrixSize >> logv.txt
done
