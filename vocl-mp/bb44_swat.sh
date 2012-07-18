gpu=1
hosts=bb44
proxy="-env PROXY_HOST_LIST=bb44"
preload="-env LD_PRELOAD=/home/liyan/vocl4.0/vocl_lib/lib/libvocl.so"
#namepub="-env MV2_NAMEPUB_DIR=/home/scxiao/workplace/port"
dpm="-env MV2_SUPPORT_DPM=1"
affinity="-env MV2_ENABLE_AFFINITY=0"
bind="-binding user:8"
#iteration_sp=(600 500 400 400 400 400)
iteration_sp=(1 1 1 1 1 1)
DIR=/home/liyan/vocl4.0/vocl/examples/swat/input
seqSize=(1 2 3 4 5 6)
#seqSize=(6 6 6 6 6 6 6)

iteration=(200 100 60 50 40 40)

for i in 0 1 2 3 4 5 
do
	echo "***************** sw = ${seqSize[i]} *****************" >> bb44_sw_result.txt
	options="$preload $dpm $namepub $affinity $proxy $bind -hosts $hosts -np 1"
	inputs="$DIR/query${seqSize[i]}K1 $DIR/sampledb${seqSize[i]}K1 5.0 0.5 ${iteration[i]} $gpu"
	echo "mpiexec.hydra $options ./swat $inputs  >> bb44_sw_result.txt"
	/home/liyan/mpich2-1.5b1/bin/mpiexec.hydra $options ./vocl_lib/bin/swat $inputs  >> bb44_sw_result.txt
done
