prog=03block
log=logs/03-block.log
#rm -r $log
run() {
    make clean
    make BLOCK_SIZE=$2 
    echo BS=$2 >> $log
    for i in {0..9}
    do
        ./matmul-$1
        python3 calculate_avg.py timing-$1.csv >> $log
        cp timing-$1.csv logs/03/timing-$1-$2-$i.csv
    done
}

for bs in {16..128..16}
do
    run $prog $bs
done
