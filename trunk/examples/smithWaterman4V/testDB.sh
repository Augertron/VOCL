DIR=../../hybrid
echo $DIR > testDBdna.log
for ((i = 1; i <= 7; i = i + 1))
do
	./swat $DIR/inputs/query"$i"K1 $DIR/inputs/testDBdna 5.0 0.5 >> testDBdna.log
	./swat $DIR/inputs/query"$i"K1 $DIR/inputs/testDBdna 5.0 0.5 >> testDBdna.log
	./swat $DIR/inputs/query"$i"K1 $DIR/inputs/testDBdna 5.0 0.5 >> testDBdna.log
done

#./swat $DIR/inputs/query1K1 $DIR/inputs/testdb1K2000 5.0 0.5 >> throughput.log 
#./swat $DIR/inputs/query1K1 $DIR/inputs/testdb1K2000 5.0 0.5 >> throughput.log 
#./swat $DIR/inputs/query1K1 $DIR/inputs/testdb1K2000 5.0 0.5 >> throughput.log 
#
#./swat $DIR/inputs/query2K1 $DIR/inputs/testdb2K1000 5.0 0.5 >> throughput.log 
#./swat $DIR/inputs/query2K1 $DIR/inputs/testdb2K1000 5.0 0.5 >> throughput.log 
#./swat $DIR/inputs/query2K1 $DIR/inputs/testdb2K1000 5.0 0.5 >> throughput.log 
#
#./swat $DIR/inputs/query3K1 $DIR/inputs/testdb3K800 5.0 0.5 >> throughput.log 
#./swat $DIR/inputs/query3K1 $DIR/inputs/testdb3K800 5.0 0.5 >> throughput.log 
#./swat $DIR/inputs/query3K1 $DIR/inputs/testdb3K800 5.0 0.5 >> throughput.log 
#
#./swat $DIR/inputs/query4K1 $DIR/inputs/testdb4K500 5.0 0.5 >> throughput.log 
#./swat $DIR/inputs/query4K1 $DIR/inputs/testdb4K500 5.0 0.5 >> throughput.log 
#./swat $DIR/inputs/query4K1 $DIR/inputs/testdb4K500 5.0 0.5 >> throughput.log 
#
#for ((i = 5; i <= 7; i = i + 1))
#do
#	./swat $DIR/inputs/query"$i"K1 $DIR/inputs/testdb"$i"K400 5.0 0.5 >> throughput.log
#	./swat $DIR/inputs/query"$i"K1 $DIR/inputs/testdb"$i"K400 5.0 0.5 >> throughput.log
#	./swat $DIR/inputs/query"$i"K1 $DIR/inputs/testdb"$i"K400 5.0 0.5 >> throughput.log
#done
