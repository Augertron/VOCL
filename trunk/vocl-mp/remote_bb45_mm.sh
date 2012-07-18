echo '' > bb45_mm_result.txt
hosts=bb45
proxy="-env PROXY_HOST_LIST=bb47"
#proxyfile="-env PROXY_HOST_FILE=hostfile0.txt"
preload="-env LD_PRELOAD=/home/liyan/vocl-ib/vocl_lib/lib/libvocl.so"
#namepub="-env MV2_NAMEPUB_DIR=/home/scxiao/workplace/port"
dpm="-env MV2_SUPPORT_DPM=1"
affinity="-env MV2_ENABLE_AFFINITY=0"
bind="-binding user:0"
#iteration_sp=(512 256 128 64 32 16)
iteration_sp=(1 1 1 1 1 1)
matrixSize=(1024 2048 3072 4096 5120 6144 7168)
#matrixSize=(3072 3072 3072 3072 3072 3072)
gpu=(1 1 1 1 1 1)
for j in 0 1 2 3 4 5 
do
	echo "***************** iter = ${iteration_sp[i]} matrix = ${matrixSize[j]} * ${matrixSize[j]} *****************" >> bb45_mm_result.txt
#	proxyfile="-env PROXY_HOST_FILE=hostfile$j.txt"
	proxyfile=""
	options="$preload $dpm $namepub $proxy -hosts $hosts $bind -np 1"
	echo "mpiexec.hydra $options ./matrixmult_sp ${matrixSize[j]} ${iteration_sp[j]} ${gpu[j]} >> bb45_mm_result.txt"
	/home/liyan/mvapich2-1.8/bin/mpiexec.hydra $options ./vocl_lib/bin/matrixmult_sp ${matrixSize[j]} ${iteration_sp[j]} ${gpu[j]} >> bb45_mm_result.txt	
done




