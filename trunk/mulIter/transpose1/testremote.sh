#for matrixSize in 1024 2048 3072 4096 5120 6144 7168 
#do
#	 ./tranO --width=$matrixSize --height=$matrixSize >> logo.txt 
#	 ./tranO --width=$matrixSize --height=$matrixSize >> logo.txt
#	 ./tranO --width=$matrixSize --height=$matrixSize >> logo.txt
#done
iteration=(200 100 100 80 80 80)
matrixSize=(1024 2048 3072 4096 5120 6144 7168)
for j in 0 1 2 3 4 5
do
	echo "mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./tranV --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt"
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./tranV --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./tranV --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./tranV --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt
done


