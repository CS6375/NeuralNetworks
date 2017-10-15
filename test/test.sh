#!/bin/bash
datasets=("phantom" "car.csv" "iris.csv" "adult.csv")

parameters=("phantom" "80 200 " "80 1000 " "80 2000 ")
parametersf=("phantom" "_it200" "_it1000" "_it2000")
neurons=("phantom"
    "3 3 2 3"
    "3 4 4 4"
    "3 10 10 10"
    "4 4 6 6 4"
    "4 5 5 5 5"
    "4 6 8 8 6"
    "4 10 10 10 10"
    "8 6 6 6 6 6 6 6 6"
    "8 10 8 8 8 8 8 8 10"
    "8 15 10 10 10 10 10 10 15"
)
neuronsf=(""
    "_layer3_3_2_3" "_layer3_4_4_4" "_layer3_10_10_10"
    "_layer4_4_6_6_4" "_layer4_5_5_5_5" "_layer4_6_8_8_6" "_layer4_10_10_10_10"
    "_layer8_6_6_6_6_6_6_6_6"
    "_layer8_10_8_8_8_8_8_8_10"
    "_layer8_15_10_10_10_10_10_10_15"
    )

mkdir ../result
for i in 1
do
    for n in {1..4}
    do
        for p in {1..3}
        do
            filename="../result/"${datasets[$i]}${parametersf[$p]}${neuronsf[$n]}.txt
            echo Execution: python3 ../src/NNDriver.py "../data/"${datasets[$i]} ${parameters[$p]} ${neurons[$n]} > $filename
            echo "" >> $filename
            python3 ../src/NNDriver.py "../data/"${datasets[$i]} ${parameters[$p]} ${neurons[$n]} >> $filename
        done
    done
done