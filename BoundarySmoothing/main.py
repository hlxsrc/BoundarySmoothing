from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np


def read_file(file_name):  # Function to read file

    f = open(file_name, "r")
    lines = f.readlines()
    return lines


def plot(self, file_name, attr1, attr2): # Function to plot data

    # read the file
    lines = read_file(file_name)

    # store configuration values
    size = lines[0].strip()
    attr = lines[1].strip()
    classes = lines[2].strip()

    A = np.loadtxt(file_name, skiprows=3, delimiter=',')

    # attribute's array
    list1 = []
    list2 = []

    plt.figure()

    # takes values from matrix
    for i in range(int(classes)):
        for j in range(int(size)):
            if A[j][int(attr)] == i:
                list1.append(A[j][attr1])
                list2.append(A[j][attr2])
        plt.scatter(list1, list2, marker='o')
        list1.clear()
        list2.clear()

    plt.title("Comparison")
    plt.xlabel("Attribute {}".format(attr1))
    plt.ylabel("Attribute {}".format(attr2))
    plt.show()


class GUI(Frame):  # Class to open txt file with GUI

    def __init__(self, parent):

        Frame.__init__(self, parent)

        self.parent = parent
        self.init_ui()

    def init_ui(self):  # Define GUI

        # Setting menu in top bar
        menu_bar = Menu(self.parent)
        self.parent.config(menu=menu_bar)

        # Setting File menu
        file_menu = Menu(menu_bar)
        menu_bar.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Open...', command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.parent.quit)

        # Setting Help menu
        help_menu = Menu(menu_bar)
        menu_bar.add_cascade(label='Help', menu=help_menu)
        help_menu.add_command(label='About')

        # To plot

        # Labels
        Label(self.parent, text="Plot: ").grid(row=0)
        Label(self.parent, text="Attribute 1").grid(row=1)
        Label(self.parent, text="Attribute 2").grid(row=2)

        # Entries
        self.parent.e1 = Entry(self.parent)
        self.parent.e2 = Entry(self.parent)

        self.parent.e1.grid(row=1, column=1, sticky=W)
        self.parent.e2.grid(row=2, column=1, sticky=W)

        a1 = self.parent.e1.get()
        a2 = self.parent.e2.get()

        # Button
        Button(self.parent, text='Show graphic', width=17,
               command=self.get_values).grid(row=3, column=1, sticky=W, pady=4)

        Label(self.parent, text=" ").grid(row=4)

        # To smoothing

        # Labels
        Label(self.parent, text="Smoothing:").grid(row=5)

        # Button
        Button(self.parent, text='Smooth Borders', width=17,
               command=self.parent.destroy).grid(row=6, column=1, sticky=W, pady=4)

        Label(self.parent, text=" ").grid(row=7)

        # Quit

        # Label
        Label(self.parent, text="Quit:").grid(row=8)

        # Button
        Button(self.parent, text='Exit', width=17,
               command=self.parent.destroy).grid(row=9, column=1, sticky=W, pady=4)

    def open_file(self):
        file_types = [('Text files', '*.txt'), ('All files', '*')]
        dlg = filedialog.Open(filetypes=file_types)
        fl = dlg.show()

        if fl != '':
            self.set_value(self, fl)

    def get_values(self):
        file_name = self.parent.file_name
        attr1 = self.parent.e1.get()
        attr2 = self.parent.e2.get()
        plot(self, file_name, int(attr1), int(attr2))

    def set_value(self, x, fn):
        self.parent.file_name = fn



def main():  # Main Program

    # Setting root
    root = Tk()

    # Setting title
    root.title('Decision Boundary Smoothing')

    # Creates a new instance
    gui = GUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
