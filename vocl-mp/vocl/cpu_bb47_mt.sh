echo ''> cpu_ib_bb47_mt_result.txt
gpu=0
hosts=bb47
proxy="-env PROXY_HOST_LIST=bb47"
preload="-env LD_PRELOAD=/home/liyan/vocl-ib/vocl_lib/lib/libvocl.so"
#namepub="-env MV2_NAMEPUB_DIR=/home/scxiao/workplace/port"
dpm="-env MV2_SUPPORT_DPM=1"
affinity="-env MV2_ENABLE_AFFINITY=0"
bind="-binding user:8"
matrixSize=(1024 2048 3072 4096 5120 6144 7168)
#matrixSize=(6144 6144 6144 6144 6144 6144 6144)

#iteration_sp=(1200 1000 800 800 800 800)
#iteration_sp=(600 500 400 400 400 400)
#iteration_sp=(1 2 4 8 16 32 64)
iteration_sp=(100 80 50 50 50 50)

#for j in 0 1 2 3 4 5 
for j in 0  
do
	echo "***************** matrix = ${matrixSize[j]} * ${matrixSize[j]} *****************" >> cpu_ib_bb47_mt_result.txt
	options="$preload $dpm $namepub $affinity $proxy $bind -hosts $hosts -np 1"
	inputs="--width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration_sp[j]} --device=$gpu"
	echo "bin/mpiexec.hydra $options ./matrixtrans_sp $inputs  >> cpu_ib_bb47_mt_result.txt"
#	/home/liyan/mvapich2-1.8/bin/mpiexec.hydra $options ./matrixtrans_sp $inputs >> cpu_ib_bb47_mt_result.txt
	/home/liyan/mvapich2-1.8/bin/mpiexec.hydra $options ./matrixtrans_sp $inputs
done
