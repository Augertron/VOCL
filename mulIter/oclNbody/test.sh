for bodynum in 15360 23040 30720 38400 46080 53760 61440

do
	./nbodyO --n=$bodynum >> logo.txt
	./nbodyO --n=$bodynum >> logo.txt
	./nbodyO --n=$bodynum >> logo.txt
done


for bodynum in 15360 23040 30720 38400 46080 53760 61440
do
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./nbodyV --n=$bodynum >> logv.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./nbodyV --n=$bodynum >> logv.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./nbodyV --n=$bodynum >> logv.txt
done
