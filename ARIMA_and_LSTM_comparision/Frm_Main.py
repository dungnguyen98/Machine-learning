import tkinter as tk
from sub_frm import coldbrew_backup as coldbrew, kentucky_derby, olympics, lebron_james, gilmore_girls

root = tk.Tk()
root.title("Time Series Forecasting")

appLabel = tk.Label(root, text="Time series forecasting - Compare ARIMA and LSTM Models", fg="#06a099", width=45)
appLabel.config(font=("Sylfaen", 20))
appLabel.grid(row=0, columnspan=3, padx=(10,10), pady=(30, 0))

from tkinter.ttk import *

style = Style()
style.configure('TButton', font= ('calibri', 20, 'bold'),borderwidth='4')

btn2 = Button(root, text='LeBron James', command=lambda : lebron_james.james_form(), width = 30)
btn2.grid(row=4, column=1, pady=10, padx=100)

btn3 = Button(root, text='Coldbrew', command=lambda : coldbrew.cold_form(), width = 30)
btn3.grid(row=5, column=1,pady = 10, padx=100)

btn4 = Button(root, text='Kentucky Derby!', command=lambda : kentucky_derby.derby_form(), width = 30)
btn4.grid(row=6, column=1, pady=10, padx=100)

btn5 = Button(root, text='Gilmore Girls', command=lambda : gilmore_girls.girl_form(), width = 30)
btn5.grid(row=8, column=1, pady=10, padx=100)

btn6 = Button(root, text='Olympics', command=lambda : olympics.olympic_form(), width = 30)
btn6.grid(row=9, column=1, pady = 10, padx=100)

root.mainloop()