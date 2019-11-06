import subprocess
import random
from tkinter import *
from tkinter import filedialog
import tkinter.ttk as ttk
import matplotlib.pyplot as plt
import numpy as np
# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import KNN from NN
from NN import KNN
# Import function numpy array to list from NN
from NN import np_to_list


# Function to read file
def read_file(file_name):

    f = open(file_name, "r")
    lines = f.readlines()
    return lines


# Function to give format to the rows that will be written in file
def list_to_string(new_list):
    new_string = ""
    # print(len(new_list))
    for i in range(len(new_list)):
        if i == len(new_list)-1:
            new_string += str(int(new_list[i])) + "\n"
        else:
            new_string += str(new_list[i]) + ","

    return new_string


# Function to find the most frequent class
def most_frequent(l):

    counter = 0
    num = l[0]

    for i in l:
        curr_frequency = l.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


# Function to find the majority of neighbors
def majority(neighbors, attr):

    # takes the class from the nearest k neighbors
    knn_classes = []
    for neighbor in neighbors:
        knn_classes.append(neighbor[int(attr)])

    # print("NEIGHBORS: ", knn_classes)
    result = most_frequent(knn_classes)

    return result


# Function to plot original data
def plot(self, file_name, attr1, attr2):

    # read the file
    lines = read_file(file_name)

    # store configuration values
    size = lines[0].strip()
    attr = lines[1].strip()
    classes = lines[2].strip()

    # stores the data after the first 3 rows
    A = np.loadtxt(file_name, skiprows=3, delimiter=',')

    # attribute's array
    list1 = []
    list2 = []

    plt.figure()

    # takes values from matrix with specific classes
    for i in range(int(classes)):
        for j in range(int(size)):
            # comparison of classes
            if A[j][int(attr)] == i:
                list1.append(A[j][attr1])
                list2.append(A[j][attr2])
        plt.scatter(list1, list2, marker='o')
        list1.clear()
        list2.clear()

    plt.title("Comparison before Boundary Softening")
    plt.xlabel("Attribute {}".format(attr1))
    plt.ylabel("Attribute {}".format(attr2))
    plt.show()

    return attr


# executes the knn algorithm in order to soften the decision boundary
def knn(file_name, attr1, attr2, num_neighbors):

    # Create new object
    nn = KNN(file_name)

    # Load dataset
    dataset = nn.load_file()

    # Defines a new record

    # read the file
    lines = read_file(file_name)

    # store configuration values
    size = lines[0].strip()
    attr = lines[1].strip()
    classes = lines[2].strip()

    # stores the data after the first 3 rows
    A = np.loadtxt(file_name, skiprows=3, delimiter=',')

    # list representing the data after the boundary smoothing
    after_boundary_smoothing = []

    # list representing excluded data
    excluded_data = []

    # for each row in A predicts the class
    # if the class from the row's k nearest neighbors is the same
    # the row will be added to a new list
    # else the row will be excluded from the new list
    # this new list is going to represent the data after the softening of the decision boundary
    for row in A:

        # predict the label
        neighbors = nn.predict_classification(dataset, row, int(num_neighbors))

        # 1. Takes the majority of the neighbors
        prediction = majority(neighbors, attr)

        if int(prediction) == int(row[int(attr)]):
            after_boundary_smoothing.append(row)
        else:
            excluded_data.append(row)

        # 2. All neighbors must have the same class
        # takes the class from the nearest k neighbors
        # knn_classes = []
        # for neighbor in neighbors:
        #     knn_classes.append(neighbor[int(attr)])
        #
        # # Finds if the element should or should not be in the new list
        # if all(x == knn_classes[0] for x in knn_classes):
        #     new_row = np_to_list(row)
        #     after_boundary_smoothing.append(new_row)
        # else:
        #     excluded_data.append(row)

    # print("Length of ABS: ", len(after_boundary_smoothing))
    # print("Length of ED: ", len(excluded_data))

    # attribute's array
    list1 = []
    list2 = []

    # takes values from matrix with specific classes
    for i in range(int(classes)):
        for row in after_boundary_smoothing:
            # comparison of classes
            if row[int(attr)] == i:
                list1.append(row[int(attr1)])
                list2.append(row[int(attr2)])
        # plot new data
        plt.scatter(list1, list2, marker='o')
        list1.clear()
        list2.clear()

    # add details to new plotted data
    plt.title("After Boundary Softening w/{} neighbors ".format(num_neighbors))
    plt.xlabel("Attribute {}".format(attr1))
    plt.ylabel("Attribute {}".format(attr2))
    plt.show()

    # returns data after boundary smoothing so it can be written to a new file
    return after_boundary_smoothing


def naive_bayes(x_train, x_test, y_train, y_test):
    # Create a Gaussian Classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(x_test)

    # returns model Accuracy, how often is the classifier correct?
    return metrics.accuracy_score(y_test, y_pred)


# Class to open txt file with GUI
class GUI(Frame):

    def __init__(self, parent):

        Frame.__init__(self, parent)

        self.parent = parent
        self.init_ui()

    # Define GUI
    def init_ui(self):

        # Setting menu in top bar
        menu_bar = Menu(self.parent)
        self.parent.config(menu=menu_bar)

        # Setting File menu
        file_menu = Menu(menu_bar)
        menu_bar.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Open...', command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label='Save As', command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.parent.quit)

        # Setting Tools menu
        tools_menu = Menu(menu_bar)
        menu_bar.add_cascade(label='Tools', menu=tools_menu)
        tools_menu.add_command(label='.txt to .arff', command=self.txt_to_arff)
        tools_menu.add_separator()
        tools_menu.add_command(label='Weka', command=self.open_weka)

        # Setting Help menu
        help_menu = Menu(menu_bar)
        menu_bar.add_cascade(label='Help', menu=help_menu)
        help_menu.add_command(label='About')

        # Text output
        self.parent.output = Text(self.parent)
        self.parent.output.grid(row=0, column=2, rowspan=17, sticky=W+E+N+S, padx=5, pady=5)

        # create a Scrollbar and associate it with output
        scroll_b = ttk.Scrollbar(self.parent, command=self.parent.output.yview)
        scroll_b.grid(row=0, column=3, rowspan=16, sticky=W+E+N+S)
        self.parent.output['yscrollcommand'] = scroll_b.set

        # Plot Labels
        Label(self.parent, text="Plot: ").grid(row=0)
        Label(self.parent, text="Attribute ").grid(row=1)
        Label(self.parent, text="Attribute ").grid(row=2)

        # Plot Entries
        self.parent.e1 = Entry(self.parent)
        self.parent.e2 = Entry(self.parent)
        self.parent.e1.grid(row=1, column=1, sticky=W)
        self.parent.e2.grid(row=2, column=1, sticky=W)

        # Plot Button
        Button(self.parent, text='Show graphic', width=17,
               command=self.plot).grid(row=3, column=1, sticky=W, pady=4)

        # Separator
        Label(self.parent, text=" ").grid(row=4)

        # Smoothing Labels
        Label(self.parent, text="Smoothing:").grid(row=5)
        Label(self.parent, text="# Neighbors").grid(row=6)

        # Smoothing Entries
        self.parent.e3 = Entry(self.parent)
        self.parent.e3.grid(row=6, column=1, sticky=W)

        # Smoothing Button
        Button(self.parent, text='Smooth Boundary', width=17,
               command=self.exec_knn).grid(row=7, column=1, sticky=W, pady=4)

        # Separator
        Label(self.parent, text=" ").grid(row=8)

        # Create test file Label
        Label(self.parent, text="Partition Test:").grid(row=9)

        # Create Button
        Button(self.parent, text='Create', width=17,
               command=self.create_test_partition).grid(row=10, column=1, sticky=W, pady=4)

        # Separator
        Label(self.parent, text=" ").grid(row=11)

        # Classification Label
        Label(self.parent, text="Classifier:").grid(row=12)

        # Initialize Tkinter variable for Radiobutton
        self.parent.v = IntVar()

        # Classifiers' radiobutton
        Radiobutton(self.parent, text="Naive Bayes", padx=20, variable=self.parent.v, value=1)\
            .grid(row=13, column=1, sticky=W, pady=4)
        Radiobutton(self.parent, text="ANN", padx=20, variable=self.parent.v, value=2)\
            .grid(row=14, column=1, sticky=W, pady=4)

        # Classify Button
        Button(self.parent, text='Classify', width=17,
               command=self.radio_choice).grid(row=15, column=1, sticky=W+E+N+S, pady=4)

    # Set file name value
    def set_file_name(self, x, fn):
        self.parent.file_name = fn

    # Set number of attributes value
    def set_num_of_attr(self, num):
        self.parent.num_of_attributes = num

    # Set original data
    def set_dataset(self, data):
        self.parent.dataset = data

    # Set dataset size value
    def set_dataset_size(self, num):
        self.parent.dataset_size = num

    # Set new data array which will be stored in a new file
    def set_new_dataset(self, data):
        self.parent.new_dataset = data

    # Set new dataset size value
    def set_new_dataset_size(self, num):
        self.parent.new_dataset_size = num

    # Set test partition
    def set_test(self, data):
        self.parent.test = data

    # Function to open a file using File Dialog
    def open_file(self):
        file_types = [('Text files', '*.txt'), ('All files', '*')]
        dlg = filedialog.Open(filetypes=file_types)
        fl = dlg.show()

        if fl != '':
            self.set_file_name(self, fl)
            data = read_file(self.parent.file_name)
            self.set_dataset(data)
            self.file_loaded()

    # Show file has been loaded correctly
    def file_loaded(self):
        # File loaded
        self.parent.output.insert(INSERT, 'File loaded... \n\n')

        # Set name of the file
        path = self.parent.file_name
        path_arr = path.split('/')
        name = path_arr[len(path_arr) - 1]
        self.parent.output.insert(INSERT, 'Name: ' + name + '\n\n')

        # Open file to get info
        file = self.parent.dataset
        size = file[0].strip()
        attr = file[1].strip()
        classes = file[2].strip()

        # Insert file info
        self.parent.output.insert(INSERT, 'Size: ' + size + '\n')
        self.parent.output.insert(INSERT, 'Attributes: ' + attr + '\n')
        self.parent.output.insert(INSERT, 'Classes: ' + classes + '\n\n')

        self.set_num_of_attr(attr)
        self.set_dataset_size(size)

    # Function to save a file
    def save_file(self):
        self.change_to_arff(self.parent.new_dataset, FALSE)

    #
    def txt_to_arff(self):
        data = self.parent.dataset
        data.pop(0)
        data.pop(0)
        data.pop(0)
        self.change_to_arff(data, TRUE)

    # txt file to arff file using filedialog
    def change_to_arff(self, dataset, flag):

        f = filedialog.asksaveasfile(mode='w', defaultextension=".arff")
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return

        # Get and write RELATION to file
        path = self.parent.file_name
        path_arr = path.split('/')
        name_arr = path_arr[len(path_arr) - 1].split('.')
        name = name_arr[0]
        relation = '@RELATION ' + name + '\n\n'
        f.write(relation)

        # Get and write ATTRIBUTES to file
        num = self.parent.num_of_attributes
        for i in range(int(num)):
            attribute = '@ATTRIBUTE ' + str(i) + ' REAL' + '\n'
            f.write(attribute)
        f.write('\n')

        data_to_save = dataset

        # Get and write DATA to file
        if flag:
            data = '@DATA\n'
            f.write(data)
            for row in data_to_save:
                f.write(row)
            f.close()
        else:
            data = '@DATA\n'
            f.write(data)
            i = 0
            for row in data_to_save:
                i = i + 1
                new_string = list_to_string(row)
                f.write(new_string)
            f.close()

    # Function to open weka.jar file
    def open_weka(self):
        subprocess.Popen(['java', '-jar', '/home/hlxs/Downloads/weka-3-8-3/weka.jar'])

    # Get values to plot original file
    def plot(self):
        file_name = self.parent.file_name
        attr1 = self.parent.e1.get()
        attr2 = self.parent.e2.get()
        plot(self, file_name, int(attr1), int(attr2))

    # Executes KNN to get data after the boundary smoothing
    def exec_knn(self):
        file_name = self.parent.file_name
        attr1 = self.parent.e1.get()
        attr2 = self.parent.e2.get()
        num_neighbors = self.parent.e3.get()
        new_data = knn(file_name, attr1, attr2, num_neighbors)
        if new_data != 0:
            self.set_new_dataset_size(len(new_data))
            self.smooth_done()
        self.set_new_dataset(new_data)

    # Show info about the smoothing process
    def smooth_done(self):
        # Task done
        self.parent.output.insert(INSERT, 'Boundary Smoothing with ' + self.parent.e3.get() + ' Neighbors Completed.\n')
        self.parent.output.insert(INSERT, '\nNew size of dataset: ' + str(self.parent.new_dataset_size) + '\n')
        excluded_data = int(self.parent.dataset_size) - int(self.parent.new_dataset_size)
        self.parent.output.insert(INSERT, 'Excluded data: ' + str(excluded_data) + '\n\n')

    # Create test partition for classifiers with size 1/4 from original size of dataset
    def create_test_partition(self):
        data = self.parent.dataset
        data.pop(0)
        data.pop(0)
        data.pop(0)
        test_size = len(data) / 4
        test = random.choices(data, k=int(test_size))
        self.set_test(test)
        self.parent.output.insert(INSERT, 'Test partition created.\n\n')

    # get choice from radio button
    def radio_choice(self):

        choice = self.parent.v.get()

        if choice == 1:
            self.naive_bayes_classifier()
        else:
            self.ann_classifier()

    def naive_bayes_classifier(self):

        x_train_original = []
        y_train_original = []
        # train data to float
        train_original = self.parent.dataset
        for row in train_original:
            np_row = np.fromstring(row, float, int(self.parent.num_of_attributes), ',')
            x_train_original.append(np_row)
            string = row.split(',')
            y_train_original.append(int(string[int(self.parent.num_of_attributes)]))

        x_train_soften = []
        y_train_soften = []
        # train data to float
        train_soften = np.asarray(self.parent.new_dataset)
        for row in train_soften:
            x_train_soften.append(np.delete(row, int(self.parent.num_of_attributes)))
            y_train_soften.append(int(row.item(int(self.parent.num_of_attributes))))

        x_train_soften = np.asarray(x_train_soften)

        x_test = []
        y_test = []
        # test data to float
        test = self.parent.test
        for row in test:
            np_row = np.fromstring(row, float, int(self.parent.num_of_attributes), ',')
            x_test.append(np_row)
            string = row.split(',')
            y_test.append(int(string[int(self.parent.num_of_attributes)]))

        acc = naive_bayes(x_train_original, x_test, y_train_original, y_test)
        self.parent.output.insert(INSERT, 'Original dataset\n')
        self.parent.output.insert(INSERT, 'Accuracy: ' + str(acc) + '\n\n')

        acc = naive_bayes(x_train_soften, x_test, y_train_soften, y_test)
        self.parent.output.insert(INSERT, 'Soften dataset\n')
        self.parent.output.insert(INSERT, 'Accuracy: ' + str(acc) + '\n\n')

    def ann_classifier(self):
        print("ANN")


# Main Program
def main():

    # Setting root
    root = Tk()

    # Setting title
    root.title('Decision Boundary Smoothing')

    # Creates a new instance
    gui = GUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
