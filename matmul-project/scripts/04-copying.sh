prog=04copying
log=logs/04-copying.log
run() {
    make clean
    make $2=1
    echo $2 >> $log
    for i in {0..9}
    do
        ./matmul-$1
        python3 calculate_avg.py timing-$1.csv >> $log
        cp timing-$1.csv logs/04/timing-$1-$2-$i.csv
    done
}

run $prog CP_KERNEL
run $prog CP_BLOCK
