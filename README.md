import tkinter as tk  # importing tkinter as we need it to create an interacrive window

# creating the main window, giving it a title and a size and making it unresizable
mainwindow = tk.Tk()
mainwindow.title("Harmonicus")
mainwindow.geometry("700x700")
mainwindow.resizable(False, False)
mainwindow.configure(bg="#222121")

# creating a greeeting label and placing it inside the main window
greetings = tk.Label(
    mainwindow, text="Welcome in Harmonicus!", font=("Arial", 20, "bold"), bg="#212121", fg="white")

greetings.place(relx=0.5, rely=0.1, anchor="center", height=50, width=500)

# creating an introductory label to the application
intro = tk.Label(
    mainwindow, text="Harmonicus was made to simulate pendulums' harmonic motion.\nHere you can choose the motion you want to simulate.",
    font=("Segoe UI - Soft and modern", 12), bg="#212121", fg="#A4A0A0")
intro.place(relx=0.5, rely=0.2, anchor="center")


# creating 3 different buttons to access the different interactions
option1 = tk.Button(mainwindow, text="Simple Pendulum simulation", font=(
    "Arial", 18), height=1, width=30, bg="#5A93FF", fg="white")
option1.place(relx=0.5, rely=0.4, anchor="center")

option2 = tk.Button(mainwindow, text="Double pendulum simulation", font=(
    "Arial", 18), height=1, width=30, bg="#34D399", fg="white")
option2.place(relx=0.5, rely=0.55, anchor="center")

option3 = tk.Button(mainwindow, text="Animations", font=(
    "Arial", 18), height=1, width=30, bg="#F472B6", fg="white")
option3.place(relx=0.5, rely=0.7, anchor="center")

# allowing the mainwindow to stay open
mainwindow.mainloop()
