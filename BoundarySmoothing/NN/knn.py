# This k-NN algorithm has been implemented by Jason Brownlee
# Taken from: Develop k-Nearest Neighbors in Python From Scratch,
#             Machine Learning Mastery
# URL: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

from csv import reader
from math import sqrt


def refactor(l):
    new_list = []
    for attr in l:
        new_list.append(float(attr))
    return new_list


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    temp_row1 = refactor(row1)
    temp_row2 = refactor(row2)
    # print("temp ROW1: ", temp_row1)
    # print("temp ROW2: ", temp_row2)
    distance = 0.0
    for i in range(len(temp_row1) - 1):
        distance += (temp_row1[i] - temp_row2[i]) ** 2
    return sqrt(distance)


class KNN:

    # Receive name of the file
    def __init__(self, filename):
        self.filename = filename

    # Load a file
    def load_file(self):
        dataset = list()
        with open(self.filename, 'r') as file:
            csv_reader = reader(file)
            file_reader = list(csv_reader)

            file_reader.pop(0)
            file_reader.pop(0)
            file_reader.pop(0)

            for row in file_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset

    # Locate the most similar neighbors
    def get_neighbors(self, train, test_row, num_neighbors):
        distances = list()
        for train_row in train:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    # Make a prediction with neighbors
    def predict_classification(self, train, test_row, num_neighbors):
        neighbors = self.get_neighbors(train, test_row, num_neighbors)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return neighbors

    # Convert string column to float
    def str_column_to_float(self, dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())

    # Convert string column to integer
    def str_column_to_int(self, dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
            # print('[%s] => %d' % (value, i))
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup
