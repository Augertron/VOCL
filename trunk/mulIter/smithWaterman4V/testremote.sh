DIR=../inputs
iteration=(0 20 20 20 20 20 20 20)

#for seqSize in 1 2 3 4 5 6 7
#do
#	echo "$seqSize, swatO"
#	./swatO $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K20 5.0 0.5 >> logO.txt
#	./swatO $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K20 5.0 0.5 >> logO.txt
#	./swatO $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K20 5.0 0.5 >> logO.txt
#done

for seqSize in 1 2 3 4 5 6
do
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt  -np 1 ./swatV $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt  -np 1 ./swatV $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -f machine.txt  -np 1 ./swatV $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
done

#for seqSize in 1 2 3 4 5 6
#do
#	mpiexec -hosts bb29,bb30 -np 1 ./swatV $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
#	mpiexec -hosts bb29,bb30 -np 1 ./swatV $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
#	mpiexec -hosts bb29,bb30 -np 1 ./swatV $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
#done


