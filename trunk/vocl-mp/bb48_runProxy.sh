dpm="-env MV2_SUPPORT_DPM=1"
#namepub="-env MV2_NAMEPUB_DIR=/home/scxiao/workplace/port"
#export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
hosts="-hosts bb48"
np="-np 1"
options="$dpm $namepub $hosts $options $np"

echo "/home/liyan/mvapich2-1.8/bin/mpiexec $options ./vocl_lib/bin/vocl_proxy"
/home/liyan/mvapich2-1.8/bin/mpiexec $options ./vocl_lib/bin/vocl_proxy

