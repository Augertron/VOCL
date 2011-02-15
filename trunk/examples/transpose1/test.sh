#for matrixSize in 1024 2048 3072 4096 5120 6144 7168
#do
#	 ./tranO --width=$matrixSize --height=$matrixSize 
#	 ./tranO --width=$matrixSize --height=$matrixSize
#	 ./tranO --width=$matrixSize --height=$matrixSize
#done

for matrixSize in 1024 2048 3072 4096 5120 6144 7168
do
	echo $matrixSize
	mpiexec -hosts bb24,bb30 -np 1 ./tranV --width=$matrixSize --height=$matrixSize 
	mpiexec -hosts bb24,bb30 -np 1 ./tranV --width=$matrixSize --height=$matrixSize
	mpiexec -hosts bb24,bb30 -np 1 ./tranV --width=$matrixSize --height=$matrixSize
done	
