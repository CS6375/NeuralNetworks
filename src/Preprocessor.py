from sys import argv

import pandas as pd


class Preprocessor:
    __NaN__ = ['?', 'NaN']

    def __init__(self, nan=None) -> None:
        if nan is not None:
            self.__NaN__ += nan

    def preprocessing(self, input_data, output_data, names=None) -> None:
        """
        Pre-process specific data set. First convert all categorical data to
        numeric, and then standardize all column except last one. Assume last
        column is always the label.
        :param input_data: DataSet to be processed
        :param output_data:
        :param names:
        :return:
        """
        df = pd.read_csv(input_data,
                         header=None,
                         names=names,
                         na_values=self.__NaN__)

        # Drop row with any NaN value.
        df = df.dropna(how='any')

        # For each column, if it is object type, cat with index and convert
        # to 'int64' type.
        for column in df:
            if df[column].dtype == 'object':
                df[column] = df[column].astype('category').cat.codes.astype(
                    'int64')

        # For each column except last one, standardize based on
        # https://en.wikipedia.org/wiki/Feature_scaling#Standardization
        for column in df.columns[:-1]:
            mean = df[column].mean()
            std = df[column].std()
            df[column] = (df[column] - mean) / std

        # Write new data frame to csv.
        df.to_csv(output_data, header=False, index=False)


def main():
    if len(argv) == 4:
        p = Preprocessor(argv[3])
    else:
        p = Preprocessor()

    p.preprocessing(argv[1], argv[2])


if __name__ == '__main__':
    main()
