import sys
import numpy as np
import pandas as pd
import math
import csv


class LinearRegression:
    def __init__(self, filename=''):
        self.filename = filename

    def train(self):
        return pd.read_csv(self.filename, header=None)

    def compute_parameter(self):
        train_data = self.train().values

        mean_y, mean_x = np.mean(train_data, axis=0)

        num = 0
        den = 0

        for Y, X in train_data:
            num += (X - mean_x) * (Y - mean_y)
            den += math.pow(X - mean_x, 2)
        m = num / den
        c = mean_y - m * mean_x
        return m, c

    def predict(self, test_data):
        prediction = []
        m, c = self.compute_parameter()
        print(m, c)

        for x0 in test_data:
            prediction.append(m * x0 + c)
        return prediction

    def store_prediction(self, predicted_values):

        f = open('output.csv', 'w', newline='\n')
        spam_writer = csv.writer(f)

        for result in predicted_values:
            spam_writer.writerow(result)
        f.close()


def main():
    train_filename = sys.argv[1]
    test_data_filename = sys.argv[2]

    linear_regression = LinearRegression(train_filename)

    test_data = pd.read_csv(test_data_filename, header=None).values

    predicted_values = linear_regression.predict(test_data)
    linear_regression.store_prediction(predicted_values)


if __name__ == '__main__':
    main()
