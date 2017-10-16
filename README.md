# NeuralNetworks

The Neural Network implemented in Python.

## Structure

The `src` directory has the following structure:

    .
    ├── LICENSE
    ├── README.md               # Current file.
    ├── data
    │   ├── adult.csv           # Standardize data set for Census Income.
    │   ├── car.csv             # Standardize data set for Car Evaluation.
    │   └── iris.csv            # Standardize data set for Iris.
    ├── result
    |   └──...                  # Test result and report.
    ├── src
    │   ├── Math.py             # Math function used in NeuralNetworks.
    │   ├── NNDriver.py         # Demonstrative driver program.
    │   ├── NeuralNetwork.py    # NeuralNetwork class implementation.
    │   ├── Perceptron.py       # Perceptron class implementation, standalone linear classifier.
    │   └── Preprocessor.py     # Preprocessor implementation.
    └── test
        └── test.sh             # Test script.

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

The default learning rate is `0.1` and the exit condition is *99% success rate
in training set* (i.e., if the neural network can correctly classify 99
percents of the training instance, it is considered to be converged.)

## Test result

The `NeuralNetwork` was tested on three data sets:

#### [Car Evaluation Data Set](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)

The best parameters so far were

    python3 NNDriver.py ../data/car.csv 50000 4 6 8 8 6

        Total training iteration executed: 3403
        Total training error = 0.9413%

        Total test error = 1.4451%

        --- Execution time: 2760.1818130016327 seconds ---

The training stop at `3403` iteration (while max iteration is specified as 50000).

For detailed analysis, refer to [Report for Car Evaluation Data Set](./result/car/report.md)

#### [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/Iris)

The best parameters so far were

    python3 NNDriver.py ../data/iris.csv 80 2000 3 4 4 4

        Total training iteration executed: 824 error: 0.8403%%
        Total training error = 0.8403%

        Total test error = 3.3333%

        --- Execution time: 18.540203094482422 seconds ---

    python3 NNDriver.py ../data/iris.csv 80 2000 3 10 10 10

        Total training iteration executed: 623 error: 0.8403%%
        Total training error = 0.8403%

        Total test error = 3.3333%

        --- Execution time: 36.852384090423584 seconds ---

Both of which occasionally reach 0% test error.

For detailed analysis, refer to [Report for Iris Data Set](./result/iris/report.md)

#### [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/Census+Income)

The best parameters so far were

    python3 NNDriver.py ../data/adult.csv 1000 4 5 5 5 5

        Training data set size: 24128 
        Testing data set size: 6033 

        Total training iteration executed: 1000
        Total training error = 15.1484% 

        Total test error = 15.3489% 

        --- Execution time: 10910.028629302979 seconds ---

For detailed analysis, refer to [Report for Census Income Data Set](./result/adult/report.md)
