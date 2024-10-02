prog=02AVX512
log=logs/02-avx512.log
run() {
    make clean
    make KERNEL_FACTOR=$2 
    echo LK=$2 >> $log
    for i in {0..10}
    do
        ./matmul-$1
        python3 calculate_avg.py timing-$1.csv >> $log
        cp timing-$1.csv logs/02/timing-$1-$2-$i.csv
    done
}

run $prog 1
#run $prog 2
#run $prog 4
#run $prog 8
#run $prog 16
