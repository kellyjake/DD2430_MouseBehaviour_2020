import tkinter as tk
from tkinter.ttk import *
from tkinter import scrolledtext , filedialog , StringVar

class CommConsole():
    def __init__(self):
        self.root = tk.Tk()
        self.label = tk.Label(text="")
        self.label.pack()
        self.update_clock()
        self.root.mainloop()
        self.root.ardile = None

    def clicked():
        res = "Welcome to " + txt.get()    
        lbl.configure(text=res)
        lbl2.configure(text=combo.get())

    def quit():
        par = chk_state.get()
        lab_txt = txt.get()
        
        self.root.quit()

    def fileparser():
        self.root.ardfile = filedialog.askopenfilename(filetypes= (("Arduino Scripts","*.ino"),("All files","*.*")))


root = tk.Tk()
root.title("Laserstuffz")
root.geometry('500x600')

lbl = tk.Label(root, text='Hello')
lbl.grid(column=0, row=0)

lbl2 = tk.Label(root, text='None')
lbl2.grid(column=1,row=1)

combo = Combobox(root)
combo['values'] = (1,2,3,4,5,'Text')
combo.current(5)
combo.grid(column=0,row=1)

chk_state = tk.BooleanVar()
chk_state.set(False)
chk = Checkbutton(root, text='Punishment', var=chk_state)
chk.grid(column=0,row=2)

txt = tk.Entry(root,width=10)
txt.grid(column=1,row=0)

scrTxt = scrolledtext.ScrolledText(root, width=40,height=10)
scrTxt.grid(column=4,row=0)

btn = tk.Button(root, text="Click here", command=clicked)
btn.grid(column=2, row=0)
txt.focus()

qbtn = tk.Button(root, text="Save and quit", command=quit)
qbtn.grid(column=0,row=3)

filebtn = tk.Button(root, text="Give .ino file", command=fileparser)
filebtn.grid(column=1,row=3)

root.mainloop()

print(chk_state.get())
print(txt.get())
print(root.ardfile)
