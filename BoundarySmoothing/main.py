import subprocess
from tkinter import *
from tkinter import filedialog  # filedialog for files
from tkinter import simpledialog  # simpledialog for user input
from tkinter import messagebox  # messagebox for showing errors
import tkinter.ttk as ttk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes model
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import train_test_split  # Import train_test_split function
from NN import KNN  # Import KNN from NN
from NN import np_to_list  # Import function numpy array to list from NN


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


# Function to check if an array is empty
def is_empty(any_structure):
    if any_structure:
        return False
    else:
        return True


# Function to plot original data
def plot(self, data, attr1, attr2, string, flag):

    # store configuration values
    attr = self.get_num_attr()
    classes = self.get_num_classes()

    # new arrays that'll have data to plot
    list1 = []
    list2 = []
    e_list1 = []
    e_list2 = []

    # Start to plot
    plt.figure()

    if not flag:
        excluded_data = np.asarray(self.get_excluded_data())
        print(excluded_data)

    # takes values from matrix with specific classes
    for i in range(int(classes)):
        for row in data:
            # comparison of classes
            if row[int(attr)] == i:
                list1.append(row[int(attr1)])
                list2.append(row[int(attr2)])
        if not flag:
            for e_row in excluded_data:
                # comparison of classes
                if e_row[int(attr)] == i:
                    e_list1.append(e_row[int(attr1)])
                    e_list2.append(e_row[int(attr2)])

        # plot new data
        plt.scatter(list1, list2, marker='o')
        if not flag:
            plt.scatter(e_list1, e_list2, marker='^')
            e_list1.clear()
            e_list2.clear()
        list1.clear()
        list2.clear()

    plt.title(format(string))
    plt.xlabel("Attribute {}".format(attr1))
    plt.ylabel("Attribute {}".format(attr2))
    plt.show()


# executes the knn algorithm in order to soften the decision boundary
def knn(self, dataset, attr1, attr2, num_neighbors):

    # Create new object
    nn = KNN()

    # store configuration values
    attr = self.get_num_attr()
    classes = self.get_num_classes()

    # list representing the data after the boundary smoothing
    after_boundary_smoothing = []

    # list representing excluded data
    excluded_data = []

    # for each row in A predicts the class
    # if the class from the row's k nearest neighbors is the same
    # the row will be added to a new list
    # else the row will be excluded from the new list
    # this new list is going to represent the data after the softening of the decision boundary
    for row in dataset:

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

    # plot after boundary smoothing
    plot(self, after_boundary_smoothing, attr1, attr2, 'Train after boundary smoothing', TRUE)

    # returns data after boundary smoothing so it can be written to a new file
    return after_boundary_smoothing, excluded_data


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

        # Setting Plot menu
        plot_menu = Menu(menu_bar)
        menu_bar.add_cascade(label='Plot', menu=plot_menu)
        plot_menu.add_command(label='Original', command=self.plot_original)
        plot_menu.add_command(label='Excluded', command=self.plot_excluded)

        # Setting Tools menu
        tools_menu = Menu(menu_bar)
        menu_bar.add_cascade(label='Tools', menu=tools_menu)
        tools_menu.add_command(label='Original .txt to .arff', command=self.original_txt_to_arff)
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
               command=self.before_smoothing).grid(row=7, column=1, sticky=W, pady=4)

        # Separator
        Label(self.parent, text=" ").grid(row=8)

        # Classification Label
        Label(self.parent, text="Classifier:").grid(row=9)

        # Initialize Tkinter variable for Radiobutton
        self.parent.v = IntVar()

        # Classifiers' radiobutton
        Radiobutton(self.parent, text="Naive Bayes", padx=20, variable=self.parent.v, value=1)\
            .grid(row=10, column=1, sticky=W, pady=4)
        Radiobutton(self.parent, text="ANN", padx=20, variable=self.parent.v, value=2)\
            .grid(row=11, column=1, sticky=W, pady=4)

        # Classify Button
        Button(self.parent, text='Classify', width=17,
               command=self.radio_choice).grid(row=12, column=1, sticky=W+E+N+S, pady=4)

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

        self.set_num_attr(attr)
        self.set_num_classes(classes)
        self.set_dataset_size(size)

        self.create_train_test_partition()

    # Create test partition for classifiers with size 1/4 from original size of dataset
    def create_train_test_partition(self):

        # Get original dataset
        dataset = self.parent.dataset
        # Takes out the first three rows
        dataset.pop(0)
        dataset.pop(0)
        dataset.pop(0)

        data = []
        target = []
        data_np = []
        for row in dataset:
            np_row = np.fromstring(row, float, int(self.get_num_attr()), ',')
            data.append(np_row)
            string = row.split(',')
            target.append(int(string[int(self.get_num_attr())]))
            new_row = np.fromstring(row, float, int(self.get_num_attr())+1, ',')
            data_np.append(new_row)

        # Split dataset into training set and test set
        x_train, x_test, y_train, y_test = train_test_split(np.asarray(data), target, test_size=0.25, random_state=109)

        self.set_train_set(x_train, y_train)
        self.set_x_train(x_train)
        self.set_y_train(y_train)
        self.set_test_set(x_test, y_test)
        self.set_x_test(x_test)
        self.set_y_test(y_test)

        self.set_train_set_size(len(x_train))
        self.set_dataset_np(np.asarray(data_np))

        self.parent.output.insert(INSERT, 'Test partition created.\n\n')
        self.parent.output.insert(INSERT, 'Size of train set: ' + str(len(x_train)) + '.\n')
        self.parent.output.insert(INSERT, 'Size of test set: ' + str(len(x_test)) + '.\n\n')

    # Function to save a file
    def save_file(self):
        self.change_to_arff(self.parent.new_train_set, FALSE)

    def plot_original(self):
        try:
            data = self.get_dataset_np()
            attr1 = simpledialog.askinteger('Plot', 'First attribute', minvalue=0)
            attr2 = simpledialog.askinteger('Plot', 'Second attribute', minvalue=0)
            plot(self, data, attr1, attr2, 'Original dataset', TRUE)
        except AttributeError:
            messagebox.showerror('Error', 'There is no data to plot')

    def plot_excluded(self):
        try:
            data = self.get_new_train_set()
            attr1 = simpledialog.askinteger('Plot', 'First attribute', minvalue=0)
            attr2 = simpledialog.askinteger('Plot', 'Second attribute', minvalue=0)
            plot(self, data, attr1, attr2, 'Excluded dataset', FALSE)
        except AttributeError:
            messagebox.showerror('Error', 'kNN for boundary smoothing has not been executed')

    # Function to change original file from txt to arrf
    def original_txt_to_arff(self):
        data = self.parent.dataset
        data.pop(0)
        data.pop(0)
        data.pop(0)
        self.change_to_arff(data, TRUE)

    # txt file to arff file using filedialog
    def change_to_arff(self, dataset, flag):

        # Open file to write in it
        f = filedialog.asksaveasfile(mode='w', defaultextension=".arff")
        if f is None:  # return `None` if dialog closed with "cancel".
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

        # Get and write DATA to file
        if flag:
            # If TRUE writes original dataset
            data = '@DATA\n'
            f.write(data)
            for row in dataset:
                f.write(row)
            f.close()
        else:
            # else writes data after boundary smoothing
            data = '@DATA\n'
            f.write(data)
            i = 0
            for row in dataset:
                i = i + 1
                new_string = list_to_string(row)
                f.write(new_string)
            f.close()

    # Function to open weka.jar file
    def open_weka(self):
        subprocess.Popen(['java', '-jar', '/home/hlxs/Downloads/weka-3-8-3/weka.jar'])

    # Get values to plot original file
    def plot(self):
        data = self.get_train_set()
        attr1 = self.parent.e1.get()
        attr2 = self.parent.e2.get()
        plot(self, data, int(attr1), int(attr2), 'Train data before boundary smoothing', TRUE)

    # TEXT before smoothing
    def before_smoothing(self):
        self.parent.output.insert(INSERT, 'Smoothing...\n\n')
        self.exec_knn()

    # Executes KNN to get data after the boundary smoothing
    def exec_knn(self):
        # Get input
        data = self.get_train_set()
        attr1 = self.parent.e1.get()
        attr2 = self.parent.e2.get()
        num_neighbors = self.parent.e3.get()

        # Execute knn
        new_data, excluded_data = knn(self, data, attr1, attr2, num_neighbors)

        # Set parameters
        if new_data != 0:
            self.set_new_train_set_size(len(new_data))
            self.smooth_done()
        self.set_new_train_set(new_data)
        self.set_excluded_data(excluded_data)

    # Show info about the smoothing process
    def smooth_done(self):
        # Task done
        self.parent.output.insert(INSERT, 'Boundary Smoothing with ' + self.parent.e3.get() + ' Neighbors Completed.\n')
        self.parent.output.insert(INSERT, '\nNew size of dataset: ' + str(self.get_new_train_set_size()) + '\n')
        excluded_data = int(self.get_train_set_size()) - int(self.get_new_train_set_size())
        self.parent.output.insert(INSERT, 'Excluded data: ' + str(excluded_data) + '\n\n')

    # Get choice from radio button
    def radio_choice(self):

        choice = self.parent.v.get()

        if choice == 1:
            self.naive_bayes_classifier()
        else:
            self.ann_classifier()

    # Naive bayes classifier
    def naive_bayes_classifier(self):

        x_train_original = self.get_x_train()
        y_train_original = self.get_y_train()

        x_train_soften = []
        y_train_soften = []
        # train data to float
        train_soften = np.asarray(self.get_new_train_set())
        for row in train_soften:
            x_train_soften.append(np.delete(row, int(self.get_num_attr())))
            y_train_soften.append(int(row.item(int(self.get_num_attr()))))

        x_train_soften = np.asarray(x_train_soften)

        x_test = self.get_x_test()
        y_test = self.get_y_test()

        acc1 = naive_bayes(x_train_original, x_test, y_train_original, y_test)
        self.parent.output.insert(INSERT, 'Original dataset\n')
        self.parent.output.insert(INSERT, 'Accuracy: ' + str(acc1) + '\n\n')

        acc2 = naive_bayes(x_train_soften, x_test, y_train_soften, y_test)
        self.parent.output.insert(INSERT, 'Soften dataset\n')
        self.parent.output.insert(INSERT, 'Accuracy: ' + str(acc2) + '\n\n')

    def ann_classifier(self):
        print("ANN")

    # GETTERS AND SETTERS
    # ------------------------------------------------------------------------

    # Set file name value
    def set_file_name(self, x, fn):
        self.parent.file_name = fn

    # Get file name value
    def get_file_name(self):
        return self.parent.file_name

    # Set original data
    def set_dataset(self, data):
        self.parent.dataset = data

    # Get original data
    def get_dataset(self):
        return self.parent.dataset

    # Set original data
    def set_dataset_np(self, data):
        self.parent.dataset_np = data

    # Get original data
    def get_dataset_np(self):
        return self.parent.dataset_np

    # Set dataset size value
    def set_dataset_size(self, num):
        self.parent.dataset_size = num

    # Get dataset size
    def get_dataset_size(self):
        return self.parent.dataset_size

    # Set number of attributes
    def set_num_attr(self, num):
        self.parent.num_attr = num

    # Get number of attributes
    def get_num_attr(self):
        return self.parent.num_attr

    # Set number of classes
    def set_num_classes(self, num):
        self.parent.num_classes = num

    # Get number of classes
    def get_num_classes(self):
        return self.parent.num_classes

    # Set train set (data and target)
    def set_train_set(self, x_train, y_train):
        # initialize new array
        train = []
        # for each row on x_train add y_train[i] value at the end
        for i, row in enumerate(x_train):
            new_row = np.append(row,y_train[i])
            train.append(new_row)

        self.parent.train = np.asarray(train)

    # Get train set (data and labels)
    def get_train_set(self):
        return self.parent.train

    # Set dataset size value
    def set_train_set_size(self, num):
        self.parent.train_size = num

    # Get dataset size
    def get_train_set_size(self):
        return self.parent.train_size

    # Set x_train (data)
    def set_x_train(self, x_train):
        self.parent.x_train = x_train

    # Get x_train (data)
    def get_x_train(self):
        return self.parent.x_train

    # Set y_train (labels)
    def set_y_train(self, y_train):
        self.parent.y_train = y_train

    # Get y_train (labels)
    def get_y_train(self):
        return self.parent.y_train

    # Set test set (data and labels)
    def set_test_set(self, x_test, y_test):
        # initialize new array
        test = []
        # for each row on x_test add y_test[i] value at the end
        for i, row in enumerate(x_test):
            new_row = np.append(row, y_test[i])
            test.append(new_row)

        self.parent.test = np.asarray(test)

    # Get test set (data and labels)
    def get_test_set(self):
        return self.parent.test

    # Set x_test (data)
    def set_x_test(self, x_test):
        self.parent.x_test = x_test

    # Get x_test (data)
    def get_x_test(self):
        return self.parent.x_test

    # Set y_test (labels)
    def set_y_test(self, y_test):
        self.parent.y_test = y_test

    # Get x_test (labels)
    def get_y_test(self):
        return self.parent.y_test

    # Set new train set after boundary smoothing
    def set_new_train_set(self, data):
        self.parent.new_train = data

    # Get new train set after boundary smoothing
    def get_new_train_set(self):
        return self.parent.new_train

    # Set new size of train set after boundary smoothing
    def set_new_train_set_size(self, num):
        self.parent.new_train_set_size = num

    # Get new size of train set after boundary smoothing
    def get_new_train_set_size(self):
        return self.parent.new_train_set_size

    # Set excluded data
    def set_excluded_data(self, data):
        self.parent.excluded_data = data

    # Get excluded dara
    def get_excluded_data(self):
        return self.parent.excluded_data

    # ------------------------------------------------------------------------

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
