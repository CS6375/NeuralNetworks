# NeuralNetworks

The Neural Network is implemented in Python 3.6.

## Structure

The `src` directory has the following structure:

    .
    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── adult.csv
    │   ├── car.csv
    │   └── iris.csv
    ├── result
    │   ├── car.csv_it1000_layer3_3_2_3.txt
    │   ├── car.csv_it2000_layer3_3_2_3.txt
    │   └── car.csv_it200_layer3_3_2_3.txt
    ├── src
    │   ├── Math.py
    │   ├── NNDriver.py
    │   ├── NeuralNetwork.py
    │   ├── Perceptron.py
    │   └── Preprocessor.py
    └── test
        └── test.sh

    5 directories, 17 files


## Compilation

The project was run on following version:

-   python 3.6.3 (the minimum version should be python 3.5)
-   numpy (1.13.3)
-   pandas (0.20.3)

No explicit compilation is needed.


## Execution

### Preprocessing

A `Preprocessor` class is defined in `Preprocessor.py` module to transform raw
data in following step:

1.  Read data from given location as `pandas` DataFrame.
2.  Convert all _categorical_ or _nominal_ data column into _numeric_ column by
    cat each value with an index value.
3.  Standardize all column except the last column, which should be the label
    column.

After preprocessing, all data column except last one would be converted into
standardized format. And last column would be in values set of `[0,1,2,...,n)`,
where `n` indicated the number of possible labels.

To use the `Preprocessor`, execute the following command:

    python3 Preprocessor.py <src-file> <dst-file>

`<src-file>` should be URL or relative/absolute location of the source file.
For example:

    python3 Preprocessor.py https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data car.csv

This command would generate file `car.csv` in the folder of the execution.

### Modeling and Testing

A `Perceptron` class was defined as the foundation of the neural network, which
is a fully function standalone linear classifier trainer.

The neural network classifier was defined as class `NeuralNetwork` in module
`NeuralNetwork.py` which offer the `train` and `test` interface. Both interface
accept list of training/testing data instance as input.

A driver program `NNDriver.py` was implemented to demonstrate basic usage of
the `NeuralNetwork`. The driver program accept normalized data set and a set of
parameters as input, split the data set, train the model and output the test
error of both training and testing data set.

To execute the driver program, execute the following command:

    python3 NNDriver.py <data-set> <split-rate> <max-iter> <hidden-layers> <vector of hidder-layer' neuron count>

For example:

    python3 NNDriver.py ../data/car.csv 80 1000 3 3 2 3

This command indicated the training set `car.csv`, split rate is `80:20`, max
iteration is `1000`, in total `3` hidden layers with `3, 2, 3` neurons in
each hidden layer.

*Note that, number of neurons of input & output layer is not needed.*

-   For input layer, number of input indicates the neurons count.
-   For output layer, possible number of output label indicates the neurons
    count.

