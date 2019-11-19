import subprocess
from tkinter import *
from tkinter import filedialog  # filedialog for files
from tkinter import simpledialog  # simpledialog for user input
from tkinter import messagebox  # messagebox for showing errors
import tkinter.ttk as ttk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes model
from sklearn.naive_bayes import MultinomialNB  # Import Gaussian Naive Bayes model
from sklearn.neural_network import MLPClassifier  # Import Neural Network model
from sklearn.model_selection import cross_val_score, cross_val_predict  # Import cross validation
from sklearn.preprocessing import StandardScaler  # Import Standar Scaler
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from NN import KNN  # Import KNN from NN
from NN import np_to_list  # Import function numpy array to list from NN


# Function to read file
def read_file(file_name):
    try:
        f = open(file_name, "r")
        lines = f.readlines()
        return lines
    except TypeError:
        messagebox.showinfo("Info", "Choose a file")


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
        knn_classes.append(neighbor[attr])

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

    # new arrays waiting for data to plot
    list1 = []
    list2 = []
    e_list1 = []
    e_list2 = []

    # Start to plot
    plt.figure()

    # if flag is false gets excluded data
    if not flag:
        excluded_data = np.asarray(self.get_excluded_data())

    # takes values from matrix with specific classes
    for i in range(int(classes)):
        for row in data:
            # comparison of classes
            if row[int(attr)] == i:
                list1.append(row[int(attr1)])
                list2.append(row[int(attr2)])
        if not flag:
            # get excluded data divided by classes
            for e_row in excluded_data:
                if e_row[int(attr)] == i:
                    e_list1.append(e_row[int(attr1)])
                    e_list2.append(e_row[int(attr2)])

        # plot new data
        plt.scatter(list1, list2, marker='o')
        if not flag:
            # plot excluded data
            plt.scatter(e_list1, e_list2, marker='^')
            e_list1.clear()
            e_list2.clear()
        list1.clear()
        list2.clear()

    # Write title and name of attributes
    plt.title(format(string))
    plt.xlabel("Attribute {}".format(attr1))
    plt.ylabel("Attribute {}".format(attr2))
    plt.show()


# executes the knn algorithm in order to soften the decision boundary
def knn(self, dataset, num_neighbors):

    # Create new object
    nn = KNN()

    # store attributes
    attr = int(self.get_num_attr())

    # list representing the data after the boundary smoothing
    after_boundary_smoothing = []

    # list representing excluded data
    excluded_data = []

    # for each row in A predicts the class
    # if the class from the row's k nearest neighbors is the same
    # the row will be added to a new list
    # else the row will be excluded from the new list
    # this new list is going to represent the data after the softening of the decision boundary
    for i, row in enumerate(dataset):

        # deletes row from dataset so it does not find itself
        aux_dataset = dataset
        aux_dataset = np.delete(aux_dataset, i, axis=0)

        # predict the label
        neighbors = nn.predict_classification(aux_dataset, row, int(num_neighbors))

        if int(num_neighbors) == 1:
            # 1.1
            if int(neighbors[0][attr]) == int(row[attr]):
                after_boundary_smoothing.append(row)
            else:
                excluded_data.append(row)

        else:
            # 1.2 Takes the majority of the neighbors
            prediction = majority(neighbors, attr)

            if int(prediction) == int(row[attr]):
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

    # returns data after boundary smoothing so it can be written to a new file
    return after_boundary_smoothing, excluded_data


# naive bayes
def naive_bayes(x_train, x_test, y_train, y_test, cv):
    # Create a Gaussian Classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    model = gnb.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(x_test)

    # model Accuracy, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # Makes cross validation
    cross_accuracy, cross_mean = cross_validation(model, x_train, y_train, cv)

    # returns info
    return accuracy, cross_accuracy, cross_mean


# Multi Layer Perceptron
def multi_layer_perceptron(x_train, x_test, y_train, y_test, momentum, learning_rate,
                           dataset_size, num_of_attr, num_of_classes, cv):

    # Get number of neurons in the hidden layer with formula Nh = Ns / (a * (Ni + No))
    # Not this one because takes dataset size as a parameter
    # hidden_layer_neurons = int(dataset_size) / 2 * (int(num_of_attr) + int(num_of_classes))

    # Get number of neurons in the hidden layer with the mean of Ni and No (N=neurons, i=input, o=output)
    hidden_layer_neurons = int(int(num_of_attr) + int(num_of_classes) / 2)

    # Normalization of data
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(x_train)

    # Now apply the transformations to the data:
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Create a Neural Network Classifier
    mlp = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=hidden_layer_neurons, momentum=momentum,
                        learning_rate='constant', learning_rate_init=learning_rate)

    # Train the model using the training sets
    model = mlp.fit(x_train, y_train)

    print(model)

    # Predict the response for test dataset
    y_pred = mlp.predict(x_test)

    # model Accuracy, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # Makes cross validation
    cross_accuracy, cross_mean = cross_validation(model, x_train, y_train, cv)

    # return info
    return accuracy, cross_accuracy, cross_mean


# cross validation
def cross_validation(model, X, y, k):

    # Perform k-fold cross validation
    scores = cross_val_score(model, X, y, cv=k)

    # Make cross validated predictions
    # predictions = cross_val_predict(model, X, y, cv=k)

    # cross validation accuracy
    accuracy = scores.mean()
    mean = scores.std() * 2

    return accuracy, mean


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

        # Smoothing Labels
        Label(self.parent, text="Training and Test:").grid(row=0)

        # Smoothing Button
        Button(self.parent, text='Create', width=17,
               command=self.create_train_test_partition).grid(row=1, column=1, sticky=W, pady=4)

        # Separator
        Label(self.parent, text=" ").grid(row=2)

        # Plot Labels
        Label(self.parent, text="Plot: ").grid(row=3)

        # Plot Button
        Button(self.parent, text='Show graphic', width=17,
               command=self.plot).grid(row=4, column=1, sticky=W, pady=4)

        # Separator
        Label(self.parent, text=" ").grid(row=5)

        # Smoothing Labels
        Label(self.parent, text="Smoothing:").grid(row=6)

        # Smoothing Button
        Button(self.parent, text='Smooth Boundary', width=17,
               command=self.exec_knn).grid(row=7, column=1, sticky=W, pady=4)

        # Plot Smoothing Button
        Button(self.parent, text='Plot', width=17,
               command=self.plot_knn).grid(row=8, column=1, sticky=W, pady=4)

        # Separator
        Label(self.parent, text=" ").grid(row=9)

        # Classification Label
        Label(self.parent, text="Classifier:").grid(row=10)

        # Initialize Tkinter variable for Radiobutton
        self.parent.v = IntVar()

        # Classifiers' radiobutton
        Radiobutton(self.parent, text="Naive Bayes", padx=2, variable=self.parent.v, value=1)\
            .grid(row=10, column=1, sticky=W, )
        Radiobutton(self.parent, text="MLP", padx=2, variable=self.parent.v, value=2)\
            .grid(row=11, column=1, sticky=W)

        # Classify Button
        Button(self.parent, text='Classify', width=17,
               command=self.radio_choice).grid(row=12, column=1, sticky=W+E+N+S, pady=4)

    # Function to open a file using File Dialog
    def open_file(self):

        try:
            file_types = [('Text files', '*.txt'), ('All files', '*')]
            dlg = filedialog.Open(filetypes=file_types)
            fl = dlg.show()

            if fl != '':
                # set file name
                self.set_file_name(self, fl)
                # read file
                data = read_file(self.parent.file_name)
                # set data
                self.set_dataset(data)
                # File loaded
                self.parent.output.insert(INSERT, 'File loaded... \n\n')
                self.file_loaded()
        except AttributeError:
            # File loaded
            self.parent.output.insert(INSERT, 'File not loaded... \n\n')

    # Show file has been loaded correctly
    def file_loaded(self):

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

    # Create test partition for classifiers with size 1/4 from original size of dataset
    def create_train_test_partition(self):
        try:
            # Get original dataset
            dataset = self.parent.dataset

            # arrays waiting for data
            data = []
            target = []
            data_np = []
            for row in dataset[3:]:
                # data columns from string to numpy array
                np_row = np.fromstring(row, float, int(self.get_num_attr()), ',')
                data.append(np_row)
                # label column from string to array (int)
                string = row.split(',')
                target.append(int(string[int(self.get_num_attr())]))
                # original data set from string to data
                new_row = np.fromstring(row, float, int(self.get_num_attr())+1, ',')
                data_np.append(new_row)

            # Split dataset into training set and test set
            x_train, x_test, y_train, y_test = train_test_split(np.asarray(data), target, test_size=0.25)

            self.set_train_set(x_train, y_train)
            self.set_x_train(x_train)
            self.set_y_train(y_train)
            self.set_test_set(x_test, y_test)
            self.set_x_test(x_test)
            self.set_y_test(y_test)

            self.set_train_set_size(len(x_train))
            self.set_dataset_np(np.asarray(data_np))

            # count occurrences
            self.parent.output.insert(INSERT, 'Number of items per class: \n\n')
            for i in range(int(self.get_num_classes())):
                self.parent.output.insert(INSERT, 'Class ' + str(i) + ': ' + str(target.count(i)) + '\n')

            self.parent.output.insert(INSERT, '\n')
            self.parent.output.insert(INSERT, '---------------------------------------------------------------------\n')

            self.parent.output.insert(INSERT, 'Train and test partition created.\n\n')
            self.parent.output.insert(INSERT, 'Size of train set: ' + str(len(x_train)) + '.\n')
            self.parent.output.insert(INSERT, 'Size of test set: ' + str(len(x_test)) + '.\n\n')

            # count occurrences
            self.parent.output.insert(INSERT, 'Number of items per class: \n\n')
            for i in range(int(self.get_num_classes())):
                self.parent.output.insert(INSERT, 'Class ' + str(i) + ': ' + str(y_train.count(i)) + '\n')

            self.parent.output.insert(INSERT, '\n')

            # Write files
            self.change_to_arff(self.get_train_set(), FALSE, FALSE, 1)
            self.change_to_arff(self.get_test_set(), FALSE, FALSE, 2)
            self.parent.output.insert(INSERT, 'Training and test files created\n\n')
            self.parent.output.insert(INSERT, '---------------------------------------------------------------------\n')

        except AttributeError:
            messagebox.showerror('Error', 'There is no data to split into training set and test set')

    # Function to save a file
    def save_file(self):
        try:
            self.change_to_arff(self.parent.new_train_set, TRUE, FALSE, 0)
        except AttributeError:
            messagebox.showerror("Error", "There is nothing to save")

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
        try:
            data = self.parent.dataset
            data.pop(0)
            data.pop(0)
            data.pop(0)
            self.change_to_arff(data, TRUE, TRUE, 0)
        except AttributeError:
            messagebox.showerror("Error", "There is nothing to save")

    # txt file to arff file using filedialog
    def change_to_arff(self, dataset, flag_ask, flag_write, flag_name):

        # get file name
        path = self.get_file_name()
        path_arr = path.split('/')
        name_arr = path_arr[len(path_arr) - 1].split('.')
        name = name_arr[0]
        new_string = ''

        if flag_ask:
            # Open file to write in it
            f = filedialog.asksaveasfile(mode='w', defaultextension=".arff")
            if f is None:  # return `None` if dialog closed with "cancel".
                return
        else:
            if flag_name == 1:
                new_string = 'data/' + name + '_training.arff'
            if flag_name == 2:
                new_string = 'data/' + name + '_test.arff'
            if flag_name == 3:
                new_string = 'data/' + name + '_softened.arff'

            f = open(new_string, 'w+')

        # Get and write RELATION to file
        relation = '@RELATION ' + name + '\n\n'
        f.write(relation)

        # Get and write ATTRIBUTES to file
        num = self.get_num_attr()
        for i in range(int(num)):
            attribute = '@ATTRIBUTE ' + str(i) + ' REAL' + '\n'
            f.write(attribute)

        string = ''
        classes = self.get_num_classes()
        for i in range(int(classes)-1):
            string += str(i) + ','
        string = string + str(int(classes)-1)
        attr_class = '@ATTRIBUTE class {' + string + '}\n\n'
        f.write(attr_class)

        # Get and write DATA to file
        if flag_write:
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
        try:
            data = self.get_train_set()
            attr1 = simpledialog.askinteger('Smoothing', 'First attribute to plot', minvalue=0)
            attr2 = simpledialog.askinteger('Smoothing', 'Second attribute to plot', minvalue=0)
            plot(self, data, int(attr1), int(attr2), 'Train data before boundary smoothing', TRUE)
        except AttributeError:
            messagebox.showerror('Error', 'There is no data to plot')
        except ValueError:
            messagebox.showerror('Error', 'The value of an attribute is wrong or does not exist')
        except IndexError:
            messagebox.showerror('Error', 'Attribute does not exist')

    # Executes KNN to get data after the boundary smoothing
    def exec_knn(self):
        try:
            # Get input
            data = self.get_train_set()

            # Get input
            num_neighbors = simpledialog.askinteger('Smoothing', 'Number of K neighbors', minvalue=0)

            # Gives info to users
            self.parent.output.insert(INSERT, 'Smoothing...\n\n')

            # Execute knn
            new_data, excluded_data = knn(self, data, num_neighbors)

            # Set parameters
            if new_data != 0:
                self.set_new_train_set_size(len(new_data))
                self.set_num_neighbors(num_neighbors)
                self.smooth_done()

            # set new train set after boundary smoothing
            self.set_new_train_set(new_data)
            # set excluded data set after boundary smoothing
            self.set_excluded_data(excluded_data)

            # reformat training set after boundary smoothing process
            x_train_soften = []
            y_train_soften = []
            train_soften = np.asarray(new_data)
            for row in train_soften:
                # change to numpy array
                x_train_soften.append(np.delete(row, int(self.get_num_attr())))
                # change labels to int
                y_train_soften.append(int(row.item(int(self.get_num_attr()))))

            # set training set after boundary smoothing process
            self.set_x_train_soft(np.asarray(x_train_soften))
            self.set_y_train_soft(y_train_soften)

            # save to file
            self.change_to_arff(new_data, FALSE, FALSE, 3)

            # count occurrences
            self.parent.output.insert(INSERT, 'Number of items per class: \n\n')
            for i in range(int(self.get_num_classes())):
                self.parent.output.insert(INSERT, 'Class ' + str(i) + ': ' + str(y_train_soften.count(i)) + '\n')
            self.parent.output.insert(INSERT, '\n')
            self.parent.output.insert(INSERT, '---------------------------------------------------------------------\n')

        # handle exceptions
        except UnboundLocalError:
            messagebox.showinfo('Info', 'Add a file in File > Open')
        except ValueError:
            messagebox.showerror('Error', 'The number of neighbors is wrong')
            # Gives info to users
            self.parent.output.insert(INSERT, 'Something went wrong\n\n')
        except AttributeError:
            messagebox.showerror('Error', 'There is no data to soft')

    # Show info about the smoothing process
    def smooth_done(self):
        # Task done
        self.parent.output.insert(INSERT, 'Boundary Smoothing with ' + str(self.get_num_neighbors()) +
                                  ' Neighbors Completed.\n')
        self.parent.output.insert(INSERT, '\nNew size of dataset: ' + str(self.get_new_train_set_size()) + '\n')
        excluded_data = int(self.get_train_set_size()) - int(self.get_new_train_set_size())
        self.parent.output.insert(INSERT, 'Excluded data: ' + str(excluded_data) + '\n\n')

    def plot_knn(self):
        # get input
        attr1 = simpledialog.askinteger('Smoothing', 'First attribute to plot', minvalue=0)
        attr2 = simpledialog.askinteger('Smoothing', 'Second attribute to plot', minvalue=0)

        # get data
        data = self.get_new_train_set()

        # plot after boundary smoothing
        plot(self, data, attr1, attr2, 'Train after boundary smoothing', TRUE)

    # Get choice from radio button and executes a classifier
    def radio_choice(self):

        choice = self.parent.v.get()

        try:
            # get original training set
            x_train_original = self.get_x_train()
            y_train_original = self.get_y_train()

            # get training set after boundary smoothing process
            x_train_soften = self.get_x_train_soft()
            y_train_soften = self.get_y_train_soft()

            # get test set
            x_test = self.get_x_test()
            y_test = self.get_y_test()

            if choice == 1:
                self.naive_bayes_classifier(x_train_original, y_train_original, x_train_soften, y_train_soften,
                                            x_test, y_test)

            if choice == 2:
                self.mlp_classifier(x_train_original, y_train_original, x_train_soften, y_train_soften,
                                    x_test, y_test)

        except AttributeError:
            messagebox.showerror("Error", 'There is no data to classify')

    # Naive bayes classifier
    def naive_bayes_classifier(self, x_train_original, y_train_original, x_train_soften, y_train_soften,
                               x_test, y_test):
        # cross validation
        cv = simpledialog.askinteger('Naive Bayes', 'Cross Validation', minvalue=2)

        # Show results of the naive bayes classifier
        self.parent.output.insert(INSERT, 'Results of the Naive Bayes Classifier\n\n')

        # get accuracy of the original training set
        acc, cross_acc, cross_mean = naive_bayes(x_train_original, x_test, y_train_original, y_test, cv)
        self.parent.output.insert(INSERT, 'Original dataset\n')
        self.parent.output.insert(INSERT, 'Accuracy with automatically generated test set: %0.2f\n' % acc)
        self.parent.output.insert(INSERT, 'Cross Validated Accuracy: %0.2f (+/- %0.2f)\n\n' % (cross_acc, cross_mean))

        # get accuracy of the softened training set
        acc, cross_acc, cross_mean = naive_bayes(x_train_soften, x_test, y_train_soften, y_test, cv)
        self.parent.output.insert(INSERT, 'Soften dataset\n')
        self.parent.output.insert(INSERT, 'Accuracy with automatically generated test set: %0.2f\n' % acc)
        self.parent.output.insert(INSERT, 'Cross Validated Accuracy: %0.2f (+/- %0.2f)\n\n' % (cross_acc, cross_mean))

        self.parent.output.insert(INSERT, '---------------------------------------------------------------------\n')

    # Multi Layer Perceptron classifier
    def mlp_classifier(self, x_train_original, y_train_original, x_train_soften, y_train_soften, x_test, y_test):

        # Get input
        momentum = simpledialog.askfloat('Multi-layer Perceptron', 'Momentum', minvalue=0)
        learning_rate = simpledialog.askfloat('Multi-layer Perceptron', 'Learning rate', minvalue=0)
        cv = simpledialog.askinteger('Multi-layer Perceptron', 'Cross Validation', minvalue=2)

        # Get number of attributes and number of classes
        num_of_attr = self.get_num_attr()
        num_of_classes = self.get_num_classes()

        # Get dataset size
        size_original = self.get_train_set_size()
        size_softened = self.get_new_train_set_size()

        # Show results of the mlp classifier
        self.parent.output.insert(INSERT, 'Results of the Multi-layer Perceptron Classifier\n\n')

        # get accuracy of the original training set
        acc, cross_acc, cross_mean = multi_layer_perceptron(x_train_original, x_test, y_train_original, y_test,
                                                              momentum, learning_rate, size_original,
                                                              num_of_attr, num_of_classes, cv)
        self.parent.output.insert(INSERT, 'Original dataset\n')
        self.parent.output.insert(INSERT, 'Accuracy with automatically generated test set: %0.2f\n' % acc)
        self.parent.output.insert(INSERT, 'Cross Validated Accuracy: %0.2f (+/- %0.2f)\n\n' % (cross_acc, cross_mean))

        # get accuracy of the softened training set
        acc, cross_acc, cross_mean = multi_layer_perceptron(x_train_soften, x_test, y_train_soften, y_test,
                                                              momentum, learning_rate, size_softened,
                                                              num_of_attr, num_of_classes, cv)
        self.parent.output.insert(INSERT, 'Soften dataset\n')
        self.parent.output.insert(INSERT, 'Accuracy with automatically generated test set: %0.2f\n' % acc)
        self.parent.output.insert(INSERT, 'Cross Validated Accuracy: %0.2f (+/- %0.2f)\n\n' % (cross_acc, cross_mean))

        self.parent.output.insert(INSERT, '---------------------------------------------------------------------\n')

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

    # Set number of neighbors
    def set_num_neighbors(self, num):
        self.parent.num_neighbors = num

    # Get number of classes
    def get_num_neighbors(self):
        return self.parent.num_neighbors

    # Set train set (data and target)
    def set_train_set(self, x_train, y_train):
        # initialize new array
        train = []
        # for each row on x_train add y_train[i] value at the end
        for i, row in enumerate(x_train):
            new_row = np.append(row, y_train[i])
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

    # Set x_train (data) after boundary smoothing
    def set_x_train_soft(self, x_train):
        self.parent.x_train_soft = x_train

    # Get x_train (data) after boundary smoothing
    def get_x_train_soft(self):
        return self.parent.x_train_soft

    # Set y_train (labels) after boundary smoothing
    def set_y_train_soft(self, y_train):
        self.parent.y_train_soft = y_train

    # Get y_train (labels) after boundary smoothing
    def get_y_train_soft(self):
        return self.parent.y_train_soft

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
