import tkinter as tk
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

# ── Physics classes ──────────────────────────────────────────────────────────


class SP:
    def __init__(self, g, L, Teta, m):
        self.g = g
        self.L = L
        self.Teta = np.radians(Teta)
        self.m = m

    def get_period(self):
        return 2 * np.pi * np.sqrt(self.L / self.g)

    def trajectory(self, t):
        """Small-angle approximation: θ(t) = θ0·cos(ωt)"""
        omega = np.sqrt(self.g / self.L)
        theta = self.Teta * np.cos(omega * t)
        x = self.L * np.sin(theta)
        y = -self.L * np.cos(theta)
        return x, y


class DP:
    def __init__(self, g, L1, L2, Teta1, Teta2, m1, m2):
        self.g = g
        self.L1 = L1
        self.L2 = L2
        self.Teta1 = np.radians(Teta1)
        self.Teta2 = np.radians(Teta2)
        self.m1 = m1
        self.m2 = m2

    def get_period(self):
        # No closed-form period for DP; return small-angle approximation
        # using the two normal-mode frequencies
        r = self.L2 / self.L1
        mu = self.m2 / (self.m1 + self.m2)
        omega_sq = (self.g / self.L1) * (1 + np.sqrt(mu * r))
        T = 2 * np.pi / np.sqrt(omega_sq)
        return T  # approximate

    def _derivs(self, state, t):
        th1, w1, th2, w2 = state
        g, L1, L2, m1, m2 = self.g, self.L1, self.L2, self.m1, self.m2
        dth = th2 - th1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(dth)**2
        den2 = (L2 / L1) * den1

        dw1 = (m2 * L1 * w1**2 * np.sin(dth) * np.cos(dth)
               + m2 * g * np.sin(th2) * np.cos(dth)
               + m2 * L2 * w2**2 * np.sin(dth)
               - (m1 + m2) * g * np.sin(th1)) / den1

        dw2 = (-m2 * L2 * w2**2 * np.sin(dth) * np.cos(dth)
               + (m1 + m2) * g * np.sin(th1) * np.cos(dth)
               - (m1 + m2) * L1 * w1**2 * np.sin(dth)
               - (m1 + m2) * g * np.sin(th2)) / den2

        return [w1, dw1, w2, dw2]

    def trajectory(self, t):
        state0 = [self.Teta1, 0, self.Teta2, 0]
        sol = odeint(self._derivs, state0, t)
        th1, th2 = sol[:, 0], sol[:, 2]
        x1 = self.L1 * np.sin(th1)
        y1 = -self.L1 * np.cos(th1)
        x2 = x1 + self.L2 * np.sin(th2)
        y2 = y1 - self.L2 * np.cos(th2)
        return x1, y1, x2, y2
    
class SHO:
    """Damped Harmonic Oscillator: m*x'' + b*x' + k*x = 0"""

    def __init__(self, k, m, x0, b=0.0):
        self.k = k
        self.m = m
        self.x0 = x0    # initial displacement from equilibrium (x0)
        self.b = b       # damping coefficient (kg/s)

    def get_period(self):
        omega0 = np.sqrt(self.k / self.m)
        gamma = self.b / (2 * self.m)
        omega_d = np.sqrt(max(omega0**2 - gamma**2, 1e-10))
        return 2 * np.pi / omega_d

    def get_regime(self):
        omega0 = np.sqrt(self.k / self.m)
        gamma = self.b / (2 * self.m)
        if np.isclose(gamma, omega0, rtol=1e-2):
            return "critically damped"
        elif gamma < omega0:
            return "under-damped"
        else:
            return "over-damped"

    def _derivs(self, state, t):
        x, v = state
        return [v, -(self.k / self.m) * x - (self.b / self.m) * v]

    def trajectory(self, t):
        state0 = [self.x0, 0]
        sol = odeint(self._derivs, state0, t)
        return sol[:, 0]
    
    def spring_coords(x_c, y_top, y_bottom, n_coils=14, width=0.12):
        margin = (y_top - y_bottom) * 0.07
        y_start = y_top  - margin
        y_end   = y_bottom + margin

        pts_y = np.linspace(y_start, y_end, n_coils * 2 + 1)
        pts_x = np.empty_like(pts_y)
        pts_x[0] = x_c
        for i in range(1, len(pts_x) - 1):
            pts_x[i] = x_c + width if i % 2 == 1 else x_c - width
        pts_x[-1] = x_c

        xs = np.concatenate([[x_c], pts_x, [x_c]])
        ys = np.concatenate([[y_top], pts_y, [y_bottom]])
        return xs, ys
 

# ── Shared window helpers ────────────────────────────────────────────────────

def make_window(title):
    mainwindow.withdraw()
    win = tk.Toplevel()
    win.title(title)
    win.geometry("900x750+{}+{}".format(
        int((win.winfo_screenwidth() - 900) / 2),
        int((win.winfo_screenheight() - 750) / 2)
    ))
    win.configure(bg="#212121")
    win.protocol("WM_DELETE_WINDOW", lambda: [
                 win.destroy(), mainwindow.deiconify()])
    return win


def back_button(win):
    tk.Button(win, text="Back to Menu", font=("Arial", 12),
              command=lambda: [win.destroy(), mainwindow.deiconify()],
              bg="#5A93FF", fg="white").place(relx=0.5, rely=0.96, anchor="center")


# ── Simple Pendulum window ───────────────────────────────────────────────────

def SP_interaction():
    win = make_window("Simple Pendulum Simulation")

    # --- Input frame (left side) ---
    input_frame = tk.Frame(win, bg="#212121")
    input_frame.place(x=20, y=20)

    fields = [("g (m/s²):", "9.81"), ("L (m):", "1"),
              ("Theta (°):", "30"),  ("m (kg):", "1")]
    entries = []
    for i, (lbl, default) in enumerate(fields):
        tk.Label(input_frame, text=lbl, bg="#212121", fg="white",
                 font=("Arial", 12)).grid(row=i, column=0, sticky="e", padx=8, pady=6)
        e = tk.Entry(input_frame, width=10, font=("Arial", 12))
        e.insert(0, default)
        e.grid(row=i, column=1, padx=8, pady=6)
        entries.append(e)

    result_label = tk.Label(win, text="", bg="#212121", fg="#00FF00",
                            font=("Arial", 12, "bold"))
    result_label.place(x=20, y=290)

    # --- Matplotlib canvas (right side) ---
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#212121")
    ax.set_facecolor("#212121")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    line,  = ax.plot([], [], 'o-', color="white", lw=2, markersize=10)
    trace, = ax.plot([], [], '-', color="#F472B6", lw=1, alpha=0.5)
    ax.plot(0, 0, 's', color="white", markersize=8)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().place(x=280, y=10, width=600, height=600)

    anim_ref = [None]
    trace_x, trace_y = [], []

    def run_animation():
        for r in anim_ref:
            if r:
                r.event_source.stop()
        trace_x.clear()
        trace_y.clear()

        try:
            g = float(entries[0].get())
            L = float(entries[1].get())
            Teta = float(entries[2].get())
            m = float(entries[3].get())
        except ValueError:
            result_label.config(text="Please enter valid numbers!")
            return

        sp = SP(g, L, Teta, m)
        T = sp.get_period()
        result_label.config(text=f"Period ≈ {T:.4f} s")

        lim = L + 0.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim - 0.2, 0.5)
        t_arr = np.linspace(0, 4 * T, 400)
        xs, ys = sp.trajectory(t_arr)

        def update(frame):
            x, y = xs[frame], ys[frame]
            line.set_data([0, x], [0, y])
            trace_x.append(x)
            trace_y.append(y)
            trace.set_data(trace_x[-200:], trace_y[-200:])
            return line, trace

        anim_ref[0] = animation.FuncAnimation(
            fig, update, frames=len(t_arr), interval=20, blit=True)
        canvas.draw()

    tk.Button(win, text="Get Period & Animate", command=run_animation,
              bg="#F472B6", fg="white", font=("Arial", 12, "bold")).place(
        x=20, y=240, width=220, height=40)

    back_button(win)


# ── Double Pendulum window ───────────────────────────────────────────────────

def DP_interaction():
    win = make_window("Double Pendulum Simulation")

    input_frame = tk.Frame(win, bg="#212121")
    input_frame.place(x=20, y=20)

    fields = [("g (m/s²):", "9.81"), ("L1 (m):", "1"), ("L2 (m):", "1"),
              ("Theta1 (°):", "90"), ("Theta2 (°):", "90"),
              ("m1 (kg):", "1"),   ("m2 (kg):", "1")]
    entries = []
    for i, (lbl, default) in enumerate(fields):
        tk.Label(input_frame, text=lbl, bg="#212121", fg="white",
                 font=("Arial", 12)).grid(row=i, column=0, sticky="e", padx=8, pady=5)
        e = tk.Entry(input_frame, width=10, font=("Arial", 12))
        e.insert(0, default)
        e.grid(row=i, column=1, padx=8, pady=5)
        entries.append(e)

    result_label = tk.Label(win, text="", bg="#212121", fg="#00FF00",
                            font=("Arial", 11, "bold"))
    result_label.place(x=20, y=430)

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#212121")
    ax.set_facecolor("#212121")
    ax.set_aspect("equal")
    ax.axis("off")
    line1, = ax.plot([], [], 'o-', color="white",   lw=2, markersize=8)
    line2, = ax.plot([], [], 'o-', color="#34D399", lw=2, markersize=10)
    trace, = ax.plot([], [], '-', color="#F472B6",  lw=1, alpha=0.6)
    ax.plot(0, 0, 's', color="white", markersize=8)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().place(x=280, y=10, width=600, height=600)

    anim_ref = [None]
    traj_data = [None]
    trace_x, trace_y = [], []

    def run_animation():
        for r in anim_ref:
            if r:
                r.event_source.stop()
        trace_x.clear()
        trace_y.clear()

        try:
            g = float(entries[0].get())
            L1 = float(entries[1].get())
            L2 = float(entries[2].get())
            Th1 = float(entries[3].get())
            Th2 = float(entries[4].get())
            m1 = float(entries[5].get())
            m2 = float(entries[6].get())
        except ValueError:
            result_label.config(text="Please enter valid numbers!")
            return

        dp = DP(g, L1, L2, Th1, Th2, m1, m2)
        T = dp.get_period()
        result_label.config(text=f"Period ≈ {T:.4f} s  (small-angle approx.)")

        lim = L1 + L2 + 0.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim - 0.2, lim * 0.4)

        t_arr = np.linspace(0, 20, 1000)
        x1, y1, x2, y2 = dp.trajectory(t_arr)
        traj_data[0] = (x1, y1, x2, y2)

        def update(frame):
            fx1, fy1, fx2, fy2 = traj_data[0]
            line1.set_data([0, fx1[frame]], [0, fy1[frame]])
            line2.set_data([fx1[frame], fx2[frame]], [fy1[frame], fy2[frame]])
            trace_x.append(fx2[frame])
            trace_y.append(fy2[frame])
            trace.set_data(trace_x[-300:], trace_y[-300:])
            return line1, line2, trace

        anim_ref[0] = animation.FuncAnimation(
            fig, update, frames=len(t_arr), interval=20, blit=True)
        canvas.draw()

    tk.Button(win, text="Get Period & Animate", command=run_animation,
              bg="#34D399", fg="white", font=("Arial", 12, "bold")).place(
        x=20, y=385, width=220, height=40)

    back_button(win)


# ── Main window ──────────────────────────────────────────────────────────────

mainwindow = tk.Tk()
mainwindow.title("Harmonicus")
width, height = 700, 700
mainwindow.geometry("{}x{}+{}+{}".format(
    width, height,
    int((mainwindow.winfo_screenwidth() - width) / 2),
    int((mainwindow.winfo_screenheight() - height) / 2)
))
mainwindow.resizable(False, False)
mainwindow.configure(bg="#212121")

tk.Label(mainwindow, text="Welcome to Harmonicus!",
         font=("Arial", 20, "bold"), bg="#212121", fg="white").place(
    relx=0.5, rely=0.1, anchor="center", height=50, width=500)

tk.Label(mainwindow,
         text="Harmonicus was made to simulate pendulums' harmonic motion.\nHere you can choose the motion you want to simulate.",
         font=("Arial", 12), bg="#212121", fg="#A4A0A0").place(
    relx=0.5, rely=0.25, anchor="center")

tk.Button(mainwindow, text="Simple Pendulum Simulation", font=("Arial", 18),
          height=1, width=30, bg="#5A93FF", fg="white",
          command=SP_interaction).place(relx=0.5, rely=0.4, anchor="center")

tk.Button(mainwindow, text="Double Pendulum Simulation", font=("Arial", 18),
          height=1, width=30, bg="#34D399", fg="white",
          command=DP_interaction).place(relx=0.5, rely=0.55, anchor="center")

tk.Button(mainwindow, text="Animations", font=("Arial", 18),
          height=1, width=30, bg="#F472B6", fg="white").place(
    relx=0.5, rely=0.7, anchor="center")

mainwindow.mainloop()

