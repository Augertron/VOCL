iteration=(200 100 100 80 80 80)
matrixSize=(1024 2048 3072 4096 5120 6144 7168)
for j in 0 1 2 3 4 5
do
	LD_PRELOAD=$HOME/vgpu/trunk/lib/libGPUv.so mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./tran --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt
	LD_PRELOAD=$HOME/vgpu/trunk/lib/libGPUv.so mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./tran --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt
	LD_PRELOAD=$HOME/vgpu/trunk/lib/libGPUv.so mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./tran --width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration[j]} >> logo.txt
done


