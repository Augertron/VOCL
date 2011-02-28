for program in BusSpeedDownload BusSpeedReadback DeviceMemory KernelCompile MaxFlops QueueDelay
do
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./$program -s 1 -d 0 >> throughput.log
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./$program -s 1 -d 1 >> throughput.log
done

