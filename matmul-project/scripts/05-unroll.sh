prog=05unroll
log=logs/05-unroll-detail.log
#rm -r $log
run() {
    make clean
    make BLOCK_SIZE=$2 UNROLL2=1
    echo BS=$2 >> $log
    for i in {0..9}
    do
        ./matmul-$1
        python3 calculate_avg.py timing-$1.csv >> $log
        cp timing-$1.csv logs/05/timing-$1-2-2-$2-$i.csv
    done
}

for bs in {32..128..16}
do
    run $prog $bs
done
