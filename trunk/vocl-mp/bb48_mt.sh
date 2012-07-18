echo ''> ib_bb48_mt_result.txt
gpu=1
hosts=bb48
proxy="-env PROXY_HOST_LIST=bb48"
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
iteration_sp=(10 10 10 10 10 10)

for j in 0 1 2 3 4 5 
do
	echo "***************** matrix = ${matrixSize[j]} * ${matrixSize[j]} *****************" >> ib_bb48_mt_result.txt
	options="$preload $dpm $namepub $affinity $proxy $bind -hosts $hosts -np 1"
	inputs="--width=${matrixSize[j]} --height=${matrixSize[j]} --iter=${iteration_sp[j]} --device=$gpu"
	echo "/home/liyan/mvapich2-1.8/bin/mpiexec.hydra $options ./matrixtrans_sp $inputs  >> ib_bb48_mt_result.txt"
	/home/liyan/mvapich2-1.8/bin/mpiexec.hydra $options ./vocl_lib/bin/matrixtrans_sp $inputs  >> ib_bb48_mt_result.txt
done
