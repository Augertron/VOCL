#DIR=../inputs
#for seqSize in 1 2 3 4 5 6 7
#do
#	echo $seqSize
#	./swato $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 >> log.txt
#	./swato $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 >> log.txt
#	./swato $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 >> log.txt
#done

DIR=../inputs
for seqSize in 1 2 3 4 5 6 7
do
	echo $seqSize
	mpiexec -hosts bb30,bb31 -np 1 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 >> logv.txt
	mpiexec -hosts bb30,bb31 -np 1 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 >> logv.txt
	mpiexec -hosts bb30,bb31 -np 1 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 >> logv.txt
done

