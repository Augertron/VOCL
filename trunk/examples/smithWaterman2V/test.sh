DIR=../../hybrid
echo $DIR
for ((i = 1; i <= 7; i = i + 1))
do
	./swat $DIR/inputs/query"$i"K1 $DIR/inputs/sampledb"$i"K1 5.0 0.5 >> pairwise.log
	./swat $DIR/inputs/query"$i"K1 $DIR/inputs/sampledb"$i"K1 5.0 0.5 >> pairwise.log
	./swat $DIR/inputs/query"$i"K1 $DIR/inputs/sampledb"$i"K1 5.0 0.5 >> pairwise.log
done


echo $DIR > testDBdna.log
for ((i = 1; i <= 7; i = i + 1))
do
	./swat $DIR/inputs/query"$i"K1 $DIR/inputs/testDBdna 5.0 0.5 >> testDBdna.log
	./swat $DIR/inputs/query"$i"K1 $DIR/inputs/testDBdna 5.0 0.5 >> testDBdna.log
	./swat $DIR/inputs/query"$i"K1 $DIR/inputs/testDBdna 5.0 0.5 >> testDBdna.log
done


