# importing tkinter as we need it to create an interactive window
import tkinter as tk


def SP_interaction():
    mainwindow.withdraw()  # Hide the main window
    SPwindow = tk.Toplevel()  # Use Toplevel instead of Tk()
    SPwindow.title("Simple Pendulum Simulation")

    # Set window size
    width = 700
    height = 700

    # Center the new window
    screen_width = SPwindow.winfo_screenwidth()
    screen_height = SPwindow.winfo_screenheight()
    center_x = int((screen_width - width) / 2)
    center_y = int((screen_height - height) / 2)

    SPwindow.geometry(f"{width}x{height}+{center_x}+{center_y}")
    SPwindow.configure(bg="#212121")

    # Add a back button to return to main menu
    back_btn = tk.Button(
        SPwindow,
        text="← Back to Menu",
        font=("Arial", 14),
        command=lambda: [SPwindow.destroy(), mainwindow.deiconify()],
        bg="#5A93FF",
        fg="white"
    )
    back_btn.place(relx=0.5, rely=0.9, anchor="center")

    # Your simple pendulum simulation code goes here


# Creating the main window
mainwindow = tk.Tk()
mainwindow.title("Harmonicus")

# Set window size and center it
width = 700
height = 700

# Get screen dimensions
screen_width = mainwindow.winfo_screenwidth()
screen_height = mainwindow.winfo_screenheight()

# Calculate center position
center_x = int((screen_width - width) / 2)
center_y = int((screen_height - height) / 2)

# Apply geometry with centering
mainwindow.geometry(f"{width}x{height}+{center_x}+{center_y}")
mainwindow.resizable(False, False)
mainwindow.configure(bg="#212121")

# Creating a greeting label
greetings = tk.Label(
    mainwindow,
    text="Welcome to Harmonicus!",
    font=("Arial", 20, "bold"),
    bg="#212121",
    fg="white"
)
greetings.place(relx=0.5, rely=0.1, anchor="center", height=50, width=500)

# Creating an introductory label
intro = tk.Label(
    mainwindow,
    text="Harmonicus was made to simulate pendulums' harmonic motion.\nHere you can choose the motion you want to simulate.",
    font=("Arial", 12),
    bg="#212121",
    fg="#A4A0A0"
)
intro.place(relx=0.5, rely=0.25, anchor="center")

# Creating 3 different buttons
option1 = tk.Button(
    mainwindow,
    text="Simple Pendulum Simulation",
    font=("Arial", 18),
    height=1,
    width=30,
    bg="#5A93FF",
    fg="white",
    command=SP_interaction
)
option1.place(relx=0.5, rely=0.4, anchor="center")

option2 = tk.Button(
    mainwindow,
    text="Double Pendulum Simulation",
    font=("Arial", 18),
    height=1,
    width=30,
    bg="#34D399",
    fg="white"
)
option2.place(relx=0.5, rely=0.55, anchor="center")

option3 = tk.Button(
    mainwindow,
    text="Animations",
    font=("Arial", 18),
    height=1,
    width=30,
    bg="#F472B6",
    fg="white"
)
option3.place(relx=0.5, rely=0.7, anchor="center")

# Make it so that the window doesn't disappear
mainwindow.mainloop()

