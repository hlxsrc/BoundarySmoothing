from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np


def read_file(file_name):  # Function to read file

    f = open(file_name, "r")
    lines = f.readlines()
    return lines


def plot(x, y):  # Function to plot data
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(x, y)

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


class OpenFile(Frame):  # Class to open txt file with GUI

    def __init__(self, parent):

        Frame.__init__(self, parent)

        self.parent = parent
        self.init_ui()

    def init_ui(self):  # Define GUI

        self.pack(fill=BOTH, expand=1)

        # Setting menu in top bar
        menu_bar = Menu(self.parent)
        self.parent.config(menu=menu_bar)

        # Setting File menu
        file_menu = Menu(menu_bar)
        menu_bar.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Open...', command=self.on_open)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.parent.quit)

        # Setting Help menu
        help_menu = Menu(menu_bar)
        menu_bar.add_cascade(label='Help', menu=help_menu)
        help_menu.add_command(label='About')

        # Setting buttons
        button_graphic = Button(self.parent, text='Show graphic', width=25, command=self.parent.destroy)
        button_smoothing = Button(self.parent, text='Smoothing', width=25, command=self.parent.destroy)
        button_graphic.pack()
        button_smoothing.pack()

    def on_open(self):

        ftypes = [('Text files', '*.txt'), ('All files', '*')]
        dlg = filedialog.Open(self, filetypes=ftypes)
        fl = dlg.show()

        if fl != '':

            # read the file
            lines = read_file(fl)

            # store configuration values
            size = lines[0].strip()
            attr = lines[1].strip()
            classes = lines[2].strip()

            A = np.loadtxt(fl, skiprows=3, delimiter=',')

            # attributes to plot given by user
            attr1 = 0
            attr2 = 1

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


def main():  # Main Program

    # Setting root
    root = Tk()

    # Setting title
    root.title('Decision Boundary Smoothing')

    # Creates a new instance
    of = OpenFile(root)
    root.mainloop()


if __name__ == '__main__':
    main()
