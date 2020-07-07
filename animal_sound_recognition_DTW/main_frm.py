import tkinter as tk
from tkinter import filedialog
from playsound import playsound
from main import *
root = tk.Tk()
root.title("Multi Media")

nameLabel = tk.Label(root, text="Enter your file path", width=40, font=("Sylfaen", 12)).grid(row=1, column=0)

entry_text = tk.StringVar()
nameEntry = tk.Entry(root, width = 50, textvariable = entry_text)
nameEntry.grid(row=1, column=1, pady=(30, 20))

resultLabel = tk.Label(root, text="Result: ", width=40, font=("Sylfaen", 12)).grid(row=6, column=0)

entry_result = tk.StringVar()
resultEntry = tk.Entry(root, width = 50, textvariable = entry_result)
resultEntry.grid(row=6, column=1, pady=(30, 20))

y_pred = None
def takeNameInput():
    global y_pred
    file_path = filedialog.askopenfilename()
    entry_text.set(file_path)
    y_pred = recognie(X_train, np.array([wav2features(file_path)]), y_train)

def display_Result():
    global y_pred
    entry_result.set(labels[int(y_pred[0])])

def play_sound():
    playsound(entry_text.get())

button = tk.Button(root, text="Choose file", command=lambda :takeNameInput())
button.grid(row=5, column=0, pady=30)

playButton = tk.Button(root, text="Play sound", command=lambda :play_sound())
playButton.grid(row=5, column=1)

displayButton = tk.Button(root, text="Display result", command=lambda :display_Result())
displayButton.grid(row=5, column=2, padx = 20)
root.mainloop()
# 44100
# 10 ms -> 1 so
# 100