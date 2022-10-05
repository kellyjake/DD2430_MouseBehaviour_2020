import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import cv2
import numpy as np
from tkinter import messagebox
import tkinter as tk

df = pd.read_excel(r'.\Analysis\20200728_behaviour2020_iv_564_1_drift_analysis.ods',engine='odf')

df2 = df.drop(labels=['Start time','Estimated time','Cam time'],axis=1)
df3 = df2.dropna()
df3

events = [i for i in dir(cv2) if 'EVENT' in i]


fileName = r'e:\ExperimentalResults\20200728\Mouse_295\20200728_behaviour2020_v_295_1\20200728_behaviour2020_v_295_1_recording_timed.avi' 











import tkinter

class simpleapp_tk(tkinter.Tk):
    def __init__(self,parent):
        tkinter.Tk.__init__(self,parent)
        self.parent = parent
        self.temp = False
        self.initialize()

    def initialize(self):
        self.geometry()
        self.geometry("500x250")
        self.bt = tkinter.Button(self,text="Bla",command=self.click)
        self.bt.place(x=5,y=5)
    def click(self):
        tkinter.messagebox.showinfo("blab","bla")

if __name__ == "__main__":
    app = simpleapp_tk(None)
    app.title('my application')
    app.mainloop()