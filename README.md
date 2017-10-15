# NeuralNetworks

The Decision Tree is implemented in Java SE 1.8.

## Structure

The `src` directory has the following structure:

    .
    ├── LICENSE
    ├── data
    │   ├── adult.csv
    │   ├── car.csv
    │   └── iris.csv
    ├── result
    │   ├── car.csv_it1000_layer3_3_2_3.txt
    │   └── car.csv_it200_layer3_3_2_3.txt
    ├── src
    │   ├── Math.py
    │   ├── NNDriver.py
    │   ├── NeuralNetwork.py
    │   ├── Perceptron.py
    │   ├── Preprocessor.py
    │   ├── __pycache__
    │   │   ├── Math.cpython-36.pyc
    │   │   ├── NeuralNetwork.cpython-36.pyc
    │   │   └── Perceptron.cpython-36.pyc
    │   ├── car45555.txt
    │   └── iris45555.txt
    └── test
        ├── a.txt
        └── test.sh

    5 directories, 18 files
    

## Compilation


## Execution

    python3 NNDriver.py ../data/car.csv 80 1000 3 3 2 3

## Design

