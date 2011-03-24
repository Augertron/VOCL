DIR=../inputs
iteration=(0 20 20 20 20 20 20 20)

for seqSize in 1 2 3 4 5 6
do
	LD_PRELOAD=$HOME/vgpu/trunk/lib/libGPUv.so mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt  -np 1 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
	LD_PRELOAD=$HOME/vgpu/trunk/lib/libGPUv.so mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt  -np 1 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
	LD_PRELOAD=$HOME/vgpu/trunk/lib/libGPUv.so mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt  -np 1 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
done


