from tkinter import *
from tkinter import filedialog


def read_file(filename):

    f = open(filename, "r")
    text = f.read()
    return text


class OpenFile(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.parent = parent
        self.init_ui()

    def init_ui(self):
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
        button_smoothing.pack();

    def on_open(self):

        ftypes = [('Text files', '*.txt'), ('All files', '*')]
        dlg = filedialog.Open(self, filetypes=ftypes)
        fl = dlg.show()

        if fl != '':
            text = read_file(fl)


def main():

    # Setting root
    root = Tk()

    # Setting title
    root.title('Decision Boundary Smoothing')

    of = OpenFile(root)
    root.mainloop()


if __name__ == '__main__':
    main()
