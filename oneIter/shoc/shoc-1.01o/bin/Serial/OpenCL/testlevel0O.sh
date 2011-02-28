for program in BusSpeedDownload BusSpeedReadback DeviceMemory KernelCompile MaxFlops QueueDelay
#for program in BusSpeedDownload BusSpeedReadback
do
	./$program -s 1 -d 0 >> throughput.log
	./$program -s 1 -d 1 >> throughput.log
done
