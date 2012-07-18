dpm="-env MV2_SUPPORT_DPM=1"
#namepub="-env MV2_NAMEPUB_DIR=/home/scxiao/workplace/port"#export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
hosts="-hosts bb45"
np="-np 1"
options="$dpm $namepub $hosts $options $np"

echo "/home/liyan/mpich2-1.5b1/bin/mpiexec $options ./vocl_lib/bin/vocl_proxy"
/home/liyan/mpich2-1.5b1/bin/mpiexec $options ./vocl_lib/bin/vocl_proxy

