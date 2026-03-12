import tkinter as tk
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

# ── Physics classes ──────────────────────────────────────────────────────────


class SP:
    def __init__(self, g, L, Teta, m):
        if g <= 0 or L <= 0 or m <= 0:
            raise ValueError("g, L, and m must be positive.")
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
        if g <= 0 or L1 <= 0 or L2 <= 0 or m1 <= 0 or m2 <= 0:
            raise ValueError("g, lengths, and masses must be positive.")
        self.g = g
        self.L1 = L1
        self.L2 = L2
        self.Teta1 = np.radians(Teta1)
        self.Teta2 = np.radians(Teta2)
        self.m1 = m1
        self.m2 = m2

    def get_period(self):
        # small-angle estimate only
        r = self.L2 / self.L1
        mu = self.m2 / (self.m1 + self.m2)
        omega_sq = (self.g / self.L1) * (1 + np.sqrt(mu * r))
        return 2 * np.pi / np.sqrt(omega_sq)

    def _derivs(self, state, t):
        th1, w1, th2, w2 = state
        g, L1, L2, m1, m2 = self.g, self.L1, self.L2, self.m1, self.m2
        dth = th2 - th1

        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(dth) ** 2
        den2 = (L2 / L1) * den1

        dw1 = (
            m2 * L1 * w1**2 * np.sin(dth) * np.cos(dth)
            + m2 * g * np.sin(th2) * np.cos(dth)
            + m2 * L2 * w2**2 * np.sin(dth)
            - (m1 + m2) * g * np.sin(th1)
        ) / den1

        dw2 = (
            -m2 * L2 * w2**2 * np.sin(dth) * np.cos(dth)
            + (m1 + m2) * g * np.sin(th1) * np.cos(dth)
            - (m1 + m2) * L1 * w1**2 * np.sin(dth)
            - (m1 + m2) * g * np.sin(th2)
        ) / den2

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
        if k <= 0 or m <= 0 or b < 0:
            raise ValueError("k and m must be positive, and b must be >= 0.")
        self.k = k
        self.m = m
        self.x0 = x0
        self.b = b

    def get_period(self):
        omega0 = np.sqrt(self.k / self.m)
        gamma = self.b / (2 * self.m)
        if gamma >= omega0:
            return None
        omega_d = np.sqrt(omega0**2 - gamma**2)
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

    @staticmethod
    def spring_coords(x_c, y_top, y_bottom, n_coils=14, width=0.12):
        margin = (y_top - y_bottom) * 0.07
        y_start = y_top - margin
        y_end = y_bottom + margin

        pts_y = np.linspace(y_start, y_end, n_coils * 2 + 1)
        pts_x = np.empty_like(pts_y)
        pts_x[0] = x_c
        for i in range(1, len(pts_x) - 1):
            pts_x[i] = x_c + width if i % 2 == 1 else x_c - width
        pts_x[-1] = x_c

        xs = np.concatenate([[x_c], pts_x, [x_c]])
        ys = np.concatenate([[y_top], pts_y, [y_bottom]])
        return xs, ys


# ── UI helpers ───────────────────────────────────────────────────────────────

def make_color_button(parent, text, bg, fg, command, x=None, y=None,
                      relx=None, rely=None, anchor=None,
                      width_chars=24, font=("Arial", 12, "bold")):
    """
    macOS-safe colored button using Label instead of Button.
    """

    btn = tk.Label(
        parent,
        text=text,
        bg=bg,
        fg=fg,
        font=font,
        width=width_chars,
        height=2,
        cursor="hand2",
        relief="flat",
        bd=0
    )

    if x is not None and y is not None:
        btn.place(x=x, y=y)
    else:
        btn.place(relx=relx, rely=rely, anchor=anchor)

    def on_enter(_):
        btn.config(bg=bg, fg=fg)

    def on_leave(_):
        btn.config(bg=bg, fg=fg)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    btn.bind("<Button-1>", lambda _: command())

    return btn


def make_window(title):
    mainwindow.withdraw()
    win = tk.Toplevel(mainwindow)
    win.title(title)
    win.geometry("900x750+{}+{}".format(
        int((win.winfo_screenwidth() - 900) / 2),
        int((win.winfo_screenheight() - 750) / 2)
    ))
    win.configure(bg="#212121")
    return win


def back_button(win, cleanup=None):
    def go_back():
        if cleanup is not None:
            cleanup()
        win.destroy()
        mainwindow.deiconify()

    make_color_button(
        win,
        text="Back to Menu",
        bg="#5A93FF",
        fg="white",
        command=go_back,
        relx=0.5,
        rely=0.96,
        anchor="center",
        width_chars=18,
        font=("Arial", 12)
    )


# ── Simple Pendulum window ───────────────────────────────────────────────────

def SP_interaction():
    win = make_window("Simple Pendulum Simulation")

    input_frame = tk.Frame(win, bg="#212121")
    input_frame.place(x=20, y=20)

    fields = [("g (m/s²):", "9.81"), ("L (m):", "1"),
              ("Theta (°):", "30"), ("m (kg):", "1")]
    entries = []
    for i, (lbl, default) in enumerate(fields):
        tk.Label(
            input_frame, text=lbl, bg="#212121", fg="white",
            font=("Arial", 12)
        ).grid(row=i, column=0, sticky="e", padx=8, pady=6)

        e = tk.Entry(input_frame, width=10, font=("Arial", 12))
        e.insert(0, default)
        e.grid(row=i, column=1, padx=8, pady=6)
        entries.append(e)

    result_label = tk.Label(
        win, text="", bg="#212121", fg="#00FF00",
        font=("Arial", 12, "bold")
    )
    result_label.place(x=20, y=290)

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#212121")
    ax.set_facecolor("#212121")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    line, = ax.plot([], [], "o-", color="white", lw=2, markersize=10)
    trace, = ax.plot([], [], "-", color="#F472B6", lw=1, alpha=0.5)
    ax.plot(0, 0, "s", color="white", markersize=8)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().place(x=280, y=10, width=600, height=600)

    anim_ref = [None]
    trace_x, trace_y = [], []

    def cleanup():
        if anim_ref[0] is not None:
            anim_ref[0].event_source.stop()
        plt.close(fig)

    def on_close():
        cleanup()
        win.destroy()
        mainwindow.deiconify()

    win.protocol("WM_DELETE_WINDOW", on_close)

    def run_animation():
        if anim_ref[0] is not None:
            anim_ref[0].event_source.stop()

        trace_x.clear()
        trace_y.clear()

        try:
            g = float(entries[0].get())
            L = float(entries[1].get())
            Teta = float(entries[2].get())
            m = float(entries[3].get())

            sp = SP(g, L, Teta, m)
        except ValueError as e:
            result_label.config(text=str(e))
            return

        T = sp.get_period()
        result_label.config(text=f"Period ≈ {T:.4f} s")

        lim = L + 0.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim - 0.2, 0.5)

        # Fixed real simulation time so parameter changes stay visible
        sim_time = 10
        fps = 50
        t_arr = np.linspace(0, sim_time, sim_time * fps)
        xs, ys = sp.trajectory(t_arr)

        def update(frame):
            x, y = xs[frame], ys[frame]
            line.set_data([0, x], [0, y])

            trace_x.append(x)
            trace_y.append(y)
            trace.set_data(trace_x[-200:], trace_y[-200:])

            return line, trace

        anim_ref[0] = animation.FuncAnimation(
            fig,
            update,
            frames=len(t_arr),
            interval=1000 / fps,
            blit=True,
            repeat=True
        )
        canvas.draw_idle()

    make_color_button(
        win,
        text="Get Period & Animate",
        bg="#F472B6",
        fg="white",
        command=run_animation,
        x=20,
        y=240,
        width_chars=22
    )

    back_button(win, cleanup)


# ── Double Pendulum window ───────────────────────────────────────────────────

def DP_interaction():
    win = make_window("Double Pendulum Simulation")

    input_frame = tk.Frame(win, bg="#212121")
    input_frame.place(x=20, y=20)

    fields = [("g (m/s²):", "9.81"), ("L1 (m):", "1"), ("L2 (m):", "1"),
              ("Theta1 (°):", "90"), ("Theta2 (°):", "90"),
              ("m1 (kg):", "1"), ("m2 (kg):", "1")]
    entries = []
    for i, (lbl, default) in enumerate(fields):
        tk.Label(
            input_frame, text=lbl, bg="#212121", fg="white",
            font=("Arial", 12)
        ).grid(row=i, column=0, sticky="e", padx=8, pady=5)

        e = tk.Entry(input_frame, width=10, font=("Arial", 12))
        e.insert(0, default)
        e.grid(row=i, column=1, padx=8, pady=5)
        entries.append(e)

    result_label = tk.Label(
        win, text="", bg="#212121", fg="#00FF00",
        font=("Arial", 11, "bold"), justify="left"
    )
    result_label.place(x=20, y=430)

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#212121")
    ax.set_facecolor("#212121")
    ax.set_aspect("equal")
    ax.axis("off")

    line1, = ax.plot([], [], "o-", color="white", lw=2, markersize=8)
    line2, = ax.plot([], [], "o-", color="#34D399", lw=2, markersize=10)
    trace, = ax.plot([], [], "-", color="#F472B6", lw=1, alpha=0.6)
    ax.plot(0, 0, "s", color="white", markersize=8)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().place(x=280, y=10, width=600, height=600)

    anim_ref = [None]
    trace_x, trace_y = [], []

    def cleanup():
        if anim_ref[0] is not None:
            anim_ref[0].event_source.stop()
        plt.close(fig)

    def on_close():
        cleanup()
        win.destroy()
        mainwindow.deiconify()

    win.protocol("WM_DELETE_WINDOW", on_close)

    def run_animation():
        if anim_ref[0] is not None:
            anim_ref[0].event_source.stop()

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

            dp = DP(g, L1, L2, Th1, Th2, m1, m2)
        except ValueError as e:
            result_label.config(text=str(e))
            return

        T = dp.get_period()
        result_label.config(text=f"Period ≈ {T:.4f} s\n(small-angle approx.)")

        lim = L1 + L2 + 0.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim - 0.2, lim * 0.4)

        # Fixed real simulation time so changes stay visible
        sim_time = 10
        fps = 50
        t_arr = np.linspace(0, sim_time, sim_time * fps)
        x1, y1, x2, y2 = dp.trajectory(t_arr)

        def update(frame):
            line1.set_data([0, x1[frame]], [0, y1[frame]])
            line2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])

            trace_x.append(x2[frame])
            trace_y.append(y2[frame])
            trace.set_data(trace_x[-300:], trace_y[-300:])

            return line1, line2, trace

        anim_ref[0] = animation.FuncAnimation(
            fig,
            update,
            frames=len(t_arr),
            interval=1000 / fps,
            blit=True,
            repeat=True
        )
        canvas.draw_idle()

    make_color_button(
        win,
        text="Get Period & Animate",
        bg="#34D399",
        fg="white",
        command=run_animation,
        x=20,
        y=385,
        width_chars=22
    )

    back_button(win, cleanup)


# ── Spring (SHO) window ──────────────────────────────────────────────────────

def SHO_interaction():
    win = make_window("Spring / Harmonic Oscillator Simulation")

    input_frame = tk.Frame(win, bg="#212121")
    input_frame.place(x=20, y=20)

    fields = [("k (N/m):", "10"), ("m (kg):", "1"),
              ("x0 (m):", "1"), ("b (kg/s):", "0")]
    entries = []
    for i, (lbl, default) in enumerate(fields):
        tk.Label(
            input_frame, text=lbl, bg="#212121", fg="white",
            font=("Arial", 12)
        ).grid(row=i, column=0, sticky="e", padx=8, pady=6)

        e = tk.Entry(input_frame, width=10, font=("Arial", 12))
        e.insert(0, default)
        e.grid(row=i, column=1, padx=8, pady=6)
        entries.append(e)

    result_label = tk.Label(
        win, text="", bg="#212121", fg="#00FF00",
        font=("Arial", 12, "bold"), justify="left"
    )
    result_label.place(x=20, y=310)

    canvas_ref = [None]
    widget_ref = [None]
    anim_ref = [None]

    def cleanup():
        if anim_ref[0] is not None:
            anim_ref[0].event_source.stop()
            anim_ref[0] = None

        if widget_ref[0] is not None:
            widget_ref[0].destroy()
            widget_ref[0] = None

        if canvas_ref[0] is not None:
            plt.close(canvas_ref[0].figure)
            canvas_ref[0] = None

    def on_close():
        cleanup()
        win.destroy()
        mainwindow.deiconify()

    win.protocol("WM_DELETE_WINDOW", on_close)

    def build_figure():
        cleanup()

        fig, (ax_sp, ax_gr) = plt.subplots(
            1, 2, figsize=(9, 5),
            facecolor="#212121",
            gridspec_kw={"width_ratios": [1, 2]}
        )

        for ax in (ax_sp, ax_gr):
            ax.set_facecolor("#212121")

        # Spring panel
        ax_sp.set_xlim(-0.5, 0.5)
        ax_sp.set_ylim(-2.2, 0.2)
        ax_sp.set_aspect("equal")
        ax_sp.axis("off")
        ax_sp.plot([-0.4, 0.4], [0, 0], color="white", lw=3)

        spr_line, = ax_sp.plot([], [], color="#FF9800", lw=2)
        mass_pt, = ax_sp.plot([], [], "s", color="#5A93FF",
                              markersize=28, zorder=5)

        # x(t) panel
        ax_gr.set_xlabel("time (s)", color="white")
        ax_gr.set_ylabel("x (m)", color="white")
        ax_gr.tick_params(colors="white")
        for spine in ax_gr.spines.values():
            spine.set_edgecolor("#555555")
        ax_gr.set_title("Displacement vs Time", color="white", fontsize=10)
        xt_ln, = ax_gr.plot([], [], color="#F472B6", lw=1.5)
        ax_gr.axhline(0, color="#555555", lw=0.8, linestyle="--")

        new_canvas = FigureCanvasTkAgg(fig, master=win)
        new_canvas.get_tk_widget().place(x=260, y=10, width=620, height=580)

        canvas_ref[0] = new_canvas
        widget_ref[0] = new_canvas.get_tk_widget()

        return fig, ax_sp, ax_gr, spr_line, mass_pt, xt_ln

    def run_animation():
        try:
            k = float(entries[0].get())
            m = float(entries[1].get())
            x0 = float(entries[2].get())
            b = float(entries[3].get())

            sho = SHO(k, m, x0, b)
        except ValueError as e:
            result_label.config(text=str(e))
            return

        T = sho.get_period()
        regime = sho.get_regime()

        if T is None:
            result_label.config(text=f"No oscillation period\n{regime}")
        else:
            result_label.config(text=f"Period ≈ {T:.4f} s\n{regime}")

        # Fixed real time so changing k visibly changes speed
        sim_time = 10
        fps = 50
        t_arr = np.linspace(0, sim_time, sim_time * fps)
        xs = sho.trajectory(t_arr)

        fig, ax_spring, ax_graph, spring_line, mass_patch, xt_line = build_figure()

        y_eq = -1.0
        y_ceil = 0.0
        amp = max(np.max(np.abs(xs)), 0.2) + 0.05

        ax_spring.set_ylim(y_eq - amp - 0.3, 0.2)
        ax_graph.set_xlim(0, t_arr[-1])
        ax_graph.set_ylim(-amp * 1.1, amp * 1.1)

        def update(frame):
            x_val = xs[frame]
            y_mass = y_eq - x_val

            sx, sy = SHO.spring_coords(
                0.0, y_ceil, y_mass + 0.14, n_coils=14, width=0.12
            )
            spring_line.set_data(sx, sy)
            mass_patch.set_data([0], [y_mass])
            xt_line.set_data(t_arr[:frame + 1], xs[:frame + 1])

            return spring_line, mass_patch, xt_line

        anim_ref[0] = animation.FuncAnimation(
            fig,
            update,
            frames=len(t_arr),
            interval=1000 / fps,
            blit=True,
            repeat=True
        )
        canvas_ref[0].draw_idle()

    make_color_button(
        win,
        text="Get Period & Animate",
        bg="#FF9800",
        fg="white",
        command=run_animation,
        x=20,
        y=260,
        width_chars=22
    )

    back_button(win, cleanup)


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

tk.Label(
    mainwindow,
    text="Welcome to Harmonicus!",
    font=("Arial", 20, "bold"),
    bg="#212121",
    fg="white"
).place(relx=0.5, rely=0.1, anchor="center", height=50, width=500)

tk.Label(
    mainwindow,
    text="Harmonicus was made to simulate pendulums' harmonic motion.\nHere you can choose the motion you want to simulate.",
    font=("Arial", 12),
    bg="#212121",
    fg="#A4A0A0"
).place(relx=0.5, rely=0.25, anchor="center")

make_color_button(
    mainwindow,
    text="Simple Pendulum Simulation",
    bg="#5A93FF",
    fg="white",
    command=SP_interaction,
    relx=0.5,
    rely=0.4,
    anchor="center",
    width_chars=30,
    font=("Arial", 18)
)

make_color_button(
    mainwindow,
    text="Double Pendulum Simulation",
    bg="#34D399",
    fg="white",
    command=DP_interaction,
    relx=0.5,
    rely=0.55,
    anchor="center",
    width_chars=30,
    font=("Arial", 18)
)

make_color_button(
    mainwindow,
    text="Spring Oscillator Simulation",
    bg="#FF9800",
    fg="white",
    command=SHO_interaction,
    relx=0.5,
    rely=0.7,
    anchor="center",
    width_chars=30,
    font=("Arial", 18)
)

mainwindow.mainloop()
