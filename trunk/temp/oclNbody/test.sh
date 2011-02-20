for $bodynum in 7680 15360 23040 30720 38400 46080 53760 61440

do
	./nbodyO --n=$bodynum >> logo.txt
	./nbodyO --n=$bodynum >> logo.txt
	./nbodyO --n=$bodynum >> logo.txt
done


for $bodynum in 7680 15360 23040 30720 38400 46080 53760 61440
do
	mpiexec -hosts gpu0031,gpu0032 -np 1 ./nbodyV --n=$bodynum >> logv.txt
	mpiexec -hosts gpu0031,gpu0032 -np 1 ./nbodyV --n=$bodynum >> logv.txt
	mpiexec -hosts gpu0031,gpu0032 -np 1 ./nbodyV --n=$bodynum >> logv.txt
done
