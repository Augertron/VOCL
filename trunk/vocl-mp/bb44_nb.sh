echo ''>> bb44_nb_result.txt
gpu=1
hosts=bb44
proxy="-env PROXY_HOST_LIST=bb44"
preload="-env LD_PRELOAD=/home/liyan/vocl4.0/vocl_lib/lib/libvocl.so"
#namepub="-env MV2_NAMEPUB_DIR=/home/scxiao/workplace/port"
dpm="-env MV2_SUPPORT_DPM=1"
affinity="-env MV2_ENABLE_AFFINITY=0"
bind="-binding user:8"
#bodynum=(15360 23040 30720 38400 46080 53760 61440)
bodynum=(53760)
#matrixSize=(6144 6144 6144 6144 6144 6144 6144)

#iteration_sp=(1200 1000 800 800 800 800)
#iteration_sp=(600 500 400 400 400 400)
iteration_sp=(20)
#iteration_sp=(1 2 4 8 16 32 64)
#iteration_sp=(1 1 1 1 1 1)

for j in 0  
do
	echo "***************** bodynum = ${bodynum[j]} *****************" >> bb44_nb_result.txt
	options="$preload $dpm $namepub $affinity $proxy $bind -hosts $hosts -np 1"
	inputs="--n=${bodynum[j]} --iter=${iteration_sp[j]} --device=$gpu"
	echo "bin/mpiexec.hydra $options ./nbody_sp $inputs  >> bb44_nb_result.txt"
	/home/liyan/mpich2-1.5b1/bin/mpiexec.hydra $options ./vocl_lib/bin/nbody_sp $inputs  >> bb44_nb_result.txt
done
