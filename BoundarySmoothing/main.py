from tkinter import *


# Setting root
root = Tk()

# Setting title
root.title('Decision Boundary Smoothing')

# Setting menu in top bar
menu = Menu(root)
root.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='Open...')
filemenu.add_separator()
filemenu.add_command(label='Exit', command=root.quit)
helpmenu = Menu(menu)
menu.add_cascade(label='Help', menu=helpmenu)
helpmenu.add_command(label='About')

# Setting buttons
buttonGraphic = Button(root, text='Show graphic', width=25, command=root.destroy)
buttonSmoothing = Button(root, text='Smoothing', width=25, command=root.destroy)
buttonGraphic.pack()
buttonSmoothing.pack();

# Main
mainloop()

