#!/usr/bin/env sh

if [ "$1" != "" ]; then
    if [ ! -d "results" ]; then
	mkdir ./results
	echo "create results directory";
    fi 
    ./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt 2>&1 | tee results/$1
else
    ./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt
fi
