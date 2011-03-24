for size in 512 1024 1536 2047 2048 4096 8192 16384 32768
do
	echo "$size"
	./bandwidth $size 30 0 >> logo.txt
	./bandwidth $size 30 0 >> logo.txt
	./bandwidth $size 30 0 >> logo.txt
done

#for size in 512 1024 1536 2047 2048 4096 8192 16384 32768
#do
#	echo "$size"
#	./bandwidth $size 30 1 >> logo.txt
#	./bandwidth $size 30 1 >> logo.txt
#	./bandwidth $size 30 1 >> logo.txt
#done

