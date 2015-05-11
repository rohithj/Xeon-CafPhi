#!/usr/bin/env sh

if [ "$1" != "" ]; then
    ./build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations 10000 2>&1 | tee results/$1
else
    ./build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations 10000
fi
