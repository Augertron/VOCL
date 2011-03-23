for size in 1 2 4 8 16 32 64
do
	echo "$size"
	mpiexec.hydra -np 1 -binding user:0 ./bandwidth $size 10 >> logo.txt
	mpiexec.hydra -np 1 -binding user:0 ./bandwidth $size 10 >> logo.txt
	mpiexec.hydra -np 1 -binding user:0 ./bandwidth $size 10 >> logo.txt
#	./bandwidth $size 10 >> logo.txt
#	./bandwidth $size 10 >> logo.txt
#	./bandwidth $size 10 >> logo.txt
done


#for bodynum in 15360 23040 30720 38400 46080 53760 61440
#do
#	mpiexec -hosts gpu0031,gpu0032 -np 1 ./nbodyV --n=$bodynum >> logv.txt
#	mpiexec -hosts gpu0031,gpu0032 -np 1 ./nbodyV --n=$bodynum >> logv.txt
#	mpiexec -hosts gpu0031,gpu0032 -np 1 ./nbodyV --n=$bodynum >> logv.txt
#done
