DIR=../inputs
iteration=(0 20 20 20 20 20 20 20)

for seqSize in 1 2 3 4 5 6
do
	echo "mpiexec.hydra -np 1 -binding user:0 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt"
	mpiexec.hydra -np 1 -binding user:0 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
	mpiexec.hydra -np 1 -binding user:0 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
	mpiexec.hydra -np 1 -binding user:0 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
done

for seqSize in 1 2 3 4 5 6
do
	echo "mpiexec.hydra -np 1 -binding user:8 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt"
	mpiexec.hydra -np 1 -binding user:8 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
	mpiexec.hydra -np 1 -binding user:8 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
	mpiexec.hydra -np 1 -binding user:8 ./swat $DIR/query"$seqSize"K1 $DIR/sampledb"$seqSize"K1 5.0 0.5 ${iteration[seqSize]} >> logO.txt
done


