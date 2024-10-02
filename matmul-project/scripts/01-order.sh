rm -r 01basic.log
run() {
    make $1=1
    echo $1 >> 01basic.log
    for i in {0..10}
    do
        ./matmul-01basic
        python3 calculate_avg.py timing-01basic.csv >> 01basic.log
        cp timing-01basic.csv logs/01/timing-01basic-$1-$i.csv
    done
}

#run IJK
#run IKJ
#run KIJ

run JIK
run JKI
run KJI
