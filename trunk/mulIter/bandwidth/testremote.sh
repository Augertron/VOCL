for size in 512 1024 2047 2048 4096 8192 16384 32768
do
	echo "$size"
	LD_PRELOAD=/home/scxiao/vgpu/trunk/lib/libGPUv.so mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./bandwidthV $size 30 0 >> logV.txt
	LD_PRELOAD=/home/scxiao/vgpu/trunk/lib/libGPUv.so mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./bandwidthV $size 30 0 >> logV.txt
	LD_PRELOAD=/home/scxiao/vgpu/trunk/lib/libGPUv.so mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./bandwidthV $size 30 0 >> logV.txt
done


