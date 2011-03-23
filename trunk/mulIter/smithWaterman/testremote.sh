DIR=../inputs

#for seqSize in 1 2 3 4 5 6 7
#do
#	echo "$seqSize, swatO"
#	./swatO $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K20 5.0 0.5 >> logO.txt
#	./swatO $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K20 5.0 0.5 >> logO.txt
#	./swatO $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K20 5.0 0.5 >> logO.txt
#done

for seqSize in 1 2 3 4 5 6 7
do
	echo "$seqSize, swatV"
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./swatV $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K20 5.0 0.5 >> logv.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./swatV $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K20 5.0 0.5 >> logv.txt
	mpiexec.hydra -env MV2_SUPPORT_DPM=1 -hosts gpu0032,gpu0031 -np 1 ./swatV $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K20 5.0 0.5 >> logv.txt
done
