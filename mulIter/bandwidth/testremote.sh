#for size in 1 2 4 8 16 32 64 128 256 512 1024 2047 2048 4096
#do
#	echo "$size"
#	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts compute-0-7,compute-0-8 -np 1 ./bandwidthV $size 1 >> logV.txt
#	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts compute-0-7,compute-0-8 -np 1 ./bandwidthV $size 1 >> logV.txt
#	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts compute-0-7,compute-0-8 -np 1 ./bandwidthV $size 1 >> logV.txt
#done

for size in 512 1024 2047 2048 4096 8192 16384 32768
do
	echo "$size"
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./bandwidthV $size 30 0 >> logV.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./bandwidthV $size 30 0 >> logV.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt -np 1 ./bandwidthV $size 30 0 >> logV.txt
done

#for size in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
#do
#	echo "$size"
#	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./bandwidthV $size 1 >> logV.txt
#	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./bandwidthV $size 1 >> logV.txt
#	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./bandwidthV $size 1 >> logV.txt
#done
#
#for size in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
#do
#	echo "$size"
#	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./bandwidthV $size 10 >> logV.txt
#	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./bandwidthV $size 10 >> logV.txt
#	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./bandwidthV $size 10 >> logV.txt
#done

#for bodynum in 15360 23040 30720 38400 46080 53760 61440
#do
#	mpiexec -hosts gpu0031,gpu0032 -np 1 ./nbodyV --n=$bodynum >> logv.txt
#	mpiexec -hosts gpu0031,gpu0032 -np 1 ./nbodyV --n=$bodynum >> logv.txt
#	mpiexec -hosts gpu0031,gpu0032 -np 1 ./nbodyV --n=$bodynum >> logv.txt
#done
