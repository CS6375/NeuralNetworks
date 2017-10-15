#!/bin/bash
datasets=("phantom" "car.csv" "iris.csv" "adult.csv")

parameters=("phantom" "80 100 " "80 200 " "80 1000 " "80 2000")

neurons=("phantom"
    "3 3 2 3"
    "3 4 4 4"
    "3 10 10 10"
    "3 20 20 20"
    "4 4 6 6 4"
    "4 5 5 5 5"
    "4 6 8 8 6"
    "4 10 10 10 10"
    "4 10 20 20 10"
    "8 6 6 6 6 6 6 6 6"
    "8 10 8 8 8 8 8 8 10"
    "8 15 10 10 10 10 10 10 15"
)
mkdir ../result
for i in 1 2
do
    for n in {1..4}
    do
        for m in {1..2}
        do
            filename="../result/"${datasets[$i]}$m$n.txt
            python3 ../src/NNDriver.py "../data/"${datasets[$i]} ${parameters[$m]} ${neurons[$n]} >> $filename
        done
    done
done