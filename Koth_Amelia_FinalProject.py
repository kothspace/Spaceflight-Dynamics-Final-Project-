"""
Amelia Koth, EP393 Final Project

Goal of project:

Curtis Problem 2.4: integrate the two-body equation (Eq. 2.22) starting
from r0, v0 and find the max altitude and when it happens. Then plot the
orbit and look at position, altitude, speed and the orbital elements vs time.

Previous tasks 1-4 were used for suplementary information.

This code requires:
    * Numpy, scipy, matplotlib (all in the .venv)

    * Use Koth_Amelia_Task3.py in the same folder

    * Results print in the terminal, figures get saved as PNGs in
      final_project_figures/

Stuff reused from my earlier tasks
  Koth_Amelia_Task3.py
        state_to_elements   r, v, orbital elements (Algorithm 4.2)
        kepler_newton       Kepler's eq with Newton (Algorithm 3.1)
        elements_to_state   elements, r, v (Algorithm 4.5)
            (last two are only used to check the RK45 answer)

  Task 2 notebook
        Earth sphere
        equal axis scaling
        the (elev, azim) views

References
  [1] Curtis, Orbital Mechanics for Engineering Students, 4th ed. (2021)
        Eq. 2.22        two-body equation of motion
        Algorithm 2.2   numerical solution of the two-body problem (p. 65)
        Algorithm 4.2   orbital elements from the state vector
  [2] scipy.integrate.solve_ivp docs (method="RK45")
  [3] youtube videos, and prior experience with RK45,
      I looked over code from my internship

I did not write the Runge-Kutta solver used Rk45 on python(scipy's). Mine is the
equations of motion function, the solver call, processing the output, and
the plots.

"""
import os # to make the output folder for the figures
import io # to capture the printed output for the pdf
import time # to time the solver runs
import contextlib # to hide the printed output from my Task 3 kepler_newton function
from math import sqrt, sin, cos, atan2, pi, degrees
import numpy as np # for arrays and linear algebra
from scipy.integrate import solve_ivp # to run the RK45 solver

# my Task 3 functions. importing Task 3 also sets matplotlib to Agg
# (saves figures to files, no windows) same as Tasks 3 and 4
from Koth_Amelia_Task3 import (MU, TWO_PI, wrap, state_to_elements,
                               kepler_newton, elements_to_state)
import matplotlib.pyplot as plt


# GIVENS from problem 2.4, Curtis p. 65

# MU = 398600 km^3/s^2 comes in from Task 3

RE = 6378.0 # Earth radius
R0 = np.array([3207.0, 5459.0, 2714.0]) # r0,km
V0 = np.array([-6.532, 0.7835, 6.142]) # v0,km/s

# SOLVER SETTINGS

N_PERIODS = 2.0 # how many orbits to run (need at least 2)
N_OUT = 2001 # output times, about one every 12 s so the plots are smoother
RTOL_FINAL = 1e-9 # rtol i ended up using
ATOL_FINAL = 1e-12 # atol i ended up using (km and km/s)

# tolerance sets to compare: (rtol, atol)
TOL_SETS = [(1e-3, 1e-6), # scipy defaults
            (1e-6, 1e-9),
            (1e-9, 1e-12), # final choice
            (1e-12, 1e-12)] # extra one to show tighter doesnt change anything
N_TIMING = 20 # one run is only a few ms so time it 20x and average

FIG_DIR = "final_project_figures" # PNGs go here

#++++++++++++++++++++
# Equations of Motion
#++++++++++++++++++++

# two_body_rates is the f(t, y) for the two-body problem, Eq. 2.22
def two_body_rates(t, y, mu):
    """
    f(t, y) for the two-body problem, Eq. 2.22.

    In : t (s) time, not actually used since gravity doesnt depend on t
         y = [x, y, z, vx, vy, vz] state (km, km/s)
         mu (km^3/s^2)
    Out: dy/dt = [vx, vy, vz, ax, ay, az]

    Just like we did in class.
    """
    r_vec = y[0:3] # position part
    v_vec = y[3:6] # velocity part
    r = sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2) # dist from Earth center
    a_vec = -mu / r**3 * r_vec # Eq 2.22  r'' = -mu r / r^3
    return np.concatenate((v_vec, a_vec)) # [r', v'] = [v, a]

# radial_velocity is the event function for the solver, it finds the times
def radial_velocity(t, y, mu):
    """
    v_r = (r . v)/r, used as the event function for the solver.

    v_r is + when going up and - when coming down, so the altitude max
    is where v_r crosses zero going + to -.

    this looks for critical points in the altitude

    """
    r_vec = y[0:3]
    v_vec = y[3:6]
    return np.dot(r_vec, v_vec) / np.linalg.norm(r_vec)


radial_velocity.direction = -1 # only + to - crossings (apogee not perigee)
radial_velocity.terminal = False # dont stop the integration at the event

#+++++++++++++++++++++
# Call Solver
#+++++++++++++++++++++

def propagate(r0, v0, t_out, rtol, atol):
    """
    Run scipy's adaptive RK45 on the orbit.

    In : r0 (km), v0 (km/s) starting state
         t_out (s) times I want output at
         rtol, atol solver tolerances
    Out: sol from solve_ivp. sol.y[0:3] is position, sol.y[3:6] is velocity
         at sol.t. sol.t_events[0] / sol.y_events[0] are the times and
         states at each altitude max.
    """
    y0 = np.concatenate((r0, v0)) # y0 = [x0 y0 z0 vx0 vy0 vz0]
    sol = solve_ivp(two_body_rates, # derivatives!!
                    (t_out[0], t_out[-1]), # t0 to tf
                    y0,
                    method="RK45",
                    t_eval=t_out, # output at these times
                    rtol=rtol, atol=atol,
                    events=radial_velocity, # finds the altitude maxes
                    args=(MU,))
    if not sol.success: # stop if the solver !failed!
        raise RuntimeError("solve_ivp failed!!!!: " + sol.message)
    return sol

#+++++++++++++++++++++
# Orbital Elements V.S. Time
#+++++++++++++++++++++

def elements_history(sol):
    """
    Run Task 3 state_to_elements at every output time.

    Out: array, one row per time, columns
         h (km^2/s), i (deg), Omega (deg), e, omega (deg), theta (deg)
    """
    n = len(sol.t)
    elems = np.zeros((n, 6)) # zero array to hold the elements at every output time
    for k in range(n): # every output time
        h, e, a, i, Om, om, th, T = state_to_elements(sol.y[0:3, k], sol.y[3:6, k], MU)
        elems[k] = [h, degrees(i), degrees(Om), e, degrees(om), degrees(th)]
    return elems

#+++++++++++++++++++++
# Check RK45
#+++++++++++++++++++++

def kepler_state(r0, v0, t):
    """
    r and v at time t from Kepler's eq instead of RK45.

    Same steps as Task 3:
    elements at t0 -> mean anomaly -> add n*t ->
    Newton for E -> true anomaly -> r and v.

    gives the time of the first apogee (where Me = pi)

    Out: r_vec (km), v_vec (km/s), t_apogee (s)
    """
    h, e, a, i, Om, om, th0, T = state_to_elements(r0, v0, MU)
    # tan(E/2) = sqrt((1-e)/(1+e)) tan(th/2).
    E0 = (2.0 * atan2(sqrt(1.0 - e) * sin(th0 / 2.0), sqrt(1.0 + e) * cos(th0 / 2.0)))
    #mean anomoly
    Me0 = E0 - e * sin(E0) # Kepler's eq, mean anomaly at t0
    n = TWO_PI / T # mean motion, rad/s
    Me = (Me0 + n * t) # mean anomaly at t

    with contextlib.redirect_stdout(io.StringIO()): # sends kepler_newton's prints to a dummy string instead of the terminal
        E, n_iter = kepler_newton(Me, e, tol=1e-12)
    theta = (2.0 * atan2(sqrt(1.0 + e) * sin(E / 2.0), sqrt(1.0 - e) * cos(E / 2.0)))
    r_vec, v_vec = elements_to_state(h, e, i, Om, om, theta, MU) # elements to r, v
    t_apogee = ((pi - Me0) / n) % T # apogee is at Me = pi
    return r_vec, v_vec, t_apogee

#+++++++++++++++++++++
# First Altitude Max
#+++++++++++++++++++++

def first_altitude_max(sol):
    """
    First altitude max after t0, found 2 ways.

    (a) solver event (v_r crosses zero + to -)
    (b) output points: first one higher than both neighbors
    Out: (t_event, alt_event, r_vec_event), (t_grid, alt_grid)
    """
    t_ev = sol.t_events[0][0] # first event time
    r_ev = sol.y_events[0][0][0:3] # position there
    alt_ev = np.linalg.norm(r_ev) - RE

    alt = np.linalg.norm(sol.y[0:3], axis=0) - RE # altitude at every output time
    k_max = None # find the first local max in the output points skip the ends
    for k in range(1, len(alt) - 1): # skip the ends
        if alt[k] > alt[k - 1] and alt[k] >= alt[k + 1]: # local max
            k_max = k # meaning the first one
            break # only want the first one
    return (t_ev, alt_ev, r_ev), (sol.t[k_max], alt[k_max])

#+++++++++++++++++++++
# Runtime Tests
#+++++++++++++++++++++

def tolerance_study(t_out, alt_exact, t_exact):
    """
    Run the same problem with each tolerance set and time it.

    Same computer, same interval, same output times every time. each one
    is run N_TIMING times and averaged since one run is only a few ms and
    the timer jumps around.
    """
    print("\nTOLERANCE AND RUNTIME COMPARISON")
    print(f"  (each runtime is the mean of {N_TIMING} runs)") #N timing is set at og
    # spacing in this is for colomn alignment, not for readability in the code
    print("   rtol     atol    f evals  runtime    max alt      time of max"
          "       alt error   time error")
    print("                              (ms)       (km)        (s)      (h)"
          "       (km)        (s)")
    # for each tolerance set, run the solver N_TIMING times and average the runtime
    for rtol, atol in TOL_SETS:
        times = []
        for n in range(N_TIMING):
            tic = time.perf_counter() # start timer
            sol = propagate(R0, V0, t_out, rtol, atol) # solver run
            times.append(time.perf_counter() - tic) # stop timer
        (t_ev, alt_ev, r_ev), grid = first_altitude_max(sol)
        print(f"  {rtol:7.0e}  {atol:7.0e}  {sol.nfev:6d}  {1000 * np.mean(times):7.2f}"
              f"  {alt_ev:11.4f}  {t_ev:9.3f}  {t_ev / 3600:8.5f}"
              f"  {alt_ev - alt_exact:+10.2e}  {t_ev - t_exact:+9.2e}")
        # so the runtime is in ms and the alt error is in km and the time error is in s, not hours

#+++++++++++++++++++++
# Plots! (my favorite part)
#+++++++++++++++++++++

BLUE, PINK, PURPLE = "#8cb7ef", "#e69af9", "#b894ed" # line colors
INK = "#0b0b0b" # markers / text


def style_axes(ax):
    """light grid. no top/right box lines"""
    ax.grid(True, color="#c4c4c4", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_orbit_3d(sol, r_max):
    """
    Earth + orbit from 3 views with equal scales.

    Earth surface, equal axes and the view list are from my Task 2
    notebook, changed so the 3 views are side by side.

    I used this method in task 2 to make the Earth sphere, and to set equal axes and the views.
    """
    x, y, z = sol.y[0], sol.y[1], sol.y[2]

    # Earth surface (Task 2): lat south pole to north pole, lon all the way around
    lat = np.linspace(-np.pi / 2, np.pi / 2, 45)
    lon = np.linspace(-np.pi, np.pi, 90)
    Lat, Lon = np.meshgrid(lat, lon, indexing="ij")
    x_e = RE * np.cos(Lat) * np.cos(Lon)
    y_e = RE * np.cos(Lat) * np.sin(Lon)
    z_e = RE * np.sin(Lat)

    # direction of motion arrows, spread out over the first orbit
    n_orbit = (len(sol.t) - 1) // int(N_PERIODS) # points in one orbit
    k_arrows = [0, n_orbit // 6, n_orbit // 3, 2 * n_orbit // 3, 5 * n_orbit // 6]

    lim = 1.1 * np.max(np.abs(sol.y[0:3])) # axis limits a little past the orbit
    # (elev, azim, title, axis pointing at the camera) for each view
    # Task 2 used azim = 45 for view 1 but this orbit is almost edge on from there
    views = [(25, -60, "View 1: Isometric view", None),
             (90, -90, "View 2: Top down view", "z"),
             (0, 0, "View 3: Side view, along x-axis", "x")]
    ticks = [-10000, 0, 10000] # just a few ticks so the labels dont overlap

    fig = plt.figure(figsize=(10.0, 3.9)) # adjusted size to make it look right
    for n, (elev, azim, title, hidden) in enumerate(views):
        ax = fig.add_subplot(1, 3, n + 1, projection="3d")
        ax.set_proj_type("ortho") # no perspective so distances arent distorted

        # Earth see-through so you can still see the orbit behind it, used this in prev. tasks
        # zorder = draw order: Earth, then orbit, arrows, markers on top
        ax.computed_zorder = False
        ax.plot_surface(x_e, y_e, z_e, color="royalblue", alpha=0.4, zorder=1)
        ax.plot(x, y, z, color=BLUE, linewidth=1.5, zorder=2, label="Orbit")
        for k in k_arrows: # direction of motion arrows
            ax.quiver(x[k], y[k], z[k], sol.y[3, k], sol.y[4, k], sol.y[5, k],
                      length=3500.0, normalize=True, color=INK,
                      linewidth=1.3, arrow_length_ratio=0.45, zorder=3)
        ax.plot([], [], [], color=INK, marker=">", linestyle="-",
                label="Direction of motion") # only here for the legend
        ax.scatter(*R0, color=PURPLE, s=45, edgecolor=INK, depthshade=False,
                   zorder=4, label="Initial position")
        ax.scatter(*r_max, color=PINK, s=70, marker="*", edgecolor=INK,
                   depthshade=False, zorder=4, label="Maximum altitude")

        # equal axis scale: same limits on all axes + cube box (Task 2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1), zoom=0.85) # equal axis scale, same as Task 2, zoom so labels dont get cut off
        ax.view_init(elev=elev, azim=azim) # position the 'camera' used in prev. tasks

                # LABEL / TITLE
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_zlabel("z (km)")
        ax.tick_params(labelsize=8)
        # this is cosmetic and makes it so numbers dont pile up on the axis when looking straight down one of them
        if hidden == "z": # looking straight down z, its labels just pile up
            ax.set_zticks([])
            ax.set_zlabel("")
        if hidden == "x":
            ax.set_xticks([])
            ax.set_xlabel("")
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels() # all 3 panels have the same legend, so just use the last one
    fig.legend(handles, labels, loc="lower center", ncol=4)
    #cosmetic
    fig.subplots_adjust(left=0.0, right=0.96, top=0.95, bottom=0.10, wspace=0.08)
    return fig


def plot_time_histories(sol, t_max, alt_max):
    """x, y, z, altitude and speed vs time"""
    t_h = sol.t / 3600.0 # hours, easier to read than seconds
    alt = np.linalg.norm(sol.y[0:3], axis=0) - RE
    speed = np.linalg.norm(sol.y[3:6], axis=0)
    # subplot to add more
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.6), sharex=True)

    axes[0].plot(t_h, sol.y[0], color=BLUE, label="x")
    axes[0].plot(t_h, sol.y[1], color=PINK, label="y")
    axes[0].plot(t_h, sol.y[2], color=PURPLE, label="z")
    axes[0].set_ylabel("Position (km)")
    axes[0].legend()

    axes[1].plot(t_h, alt, color=BLUE)
    axes[1].plot(t_max / 3600.0, alt_max, "^", color=PINK, markersize=9,
                 label=f"First max: {alt_max:.1f} km at {t_max / 3600:.3f} h")
    axes[1].set_ylabel("Altitude (km)")
    axes[1].set_ylim(0.0, 1.25 * alt_max)
    axes[1].legend(loc="upper center")

    axes[2].plot(t_h, speed, color=BLUE)
    axes[2].set_ylabel("Speed (km/s)")
    axes[2].set_xlabel("Time after $t_0$ (h)")

    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    return fig


def plot_elements(sol, elems):
    """all 6 orbital elements vs time"""
    t_h = sol.t / 3600.0
    # column, y label
    panels = [(0, "$h$ (km$^2$/s)"), (3, "$e$"),
              (1, "$i$ (deg)"), (2, r"$\Omega$ (deg)"),
              (4, r"$\omega$ (deg)"), (5, r"$\theta$ (deg)")] # theta isnt constant

    fig, axes = plt.subplots(3, 2, figsize=(7.5, 6.6), sharex=True)
    for ax, (col, ylabel) in zip(axes.flat, panels):
        ax.plot(t_h, elems[:, col], color=BLUE)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if col != 5: # constants: small window around the starting value
            ax.set_ylim(0.999 * elems[0, col], 1.001 * elems[0, col])
            ax.ticklabel_format(axis="y", useOffset=False)
    axes[2, 0].set_xlabel("Time after $t_0$ (h)")
    axes[2, 1].set_xlabel("Time after $t_0$ (h)")
    fig.tight_layout()
    return fig

#++++++++++++++++++++++
# Main
#++++++++++++++++++++++

def main():
    print("EP393 Final Project : Curtis Problem 2.4, two-body orbit with RK45")

    # elements at t0 (Task 3 function). need T for the time span
    h, e, a, i, Om, om, th0, T = state_to_elements(R0, V0, MU)
    t_out = np.linspace(0.0, N_PERIODS * T, N_OUT) # output times, t0 = 0

    # print the givens
    print("\nINITIAL STATE (t0)")
    print(f"  r0 = [{R0[0]:10.3f} {R0[1]:10.3f} {R0[2]:10.3f}] km")
    print(f"  v0 = [{V0[0]:10.5f} {V0[1]:10.5f} {V0[2]:10.5f}] km/s")
    print(f"  altitude = {np.linalg.norm(R0) - RE:.3f} km,"
          f" speed = {np.linalg.norm(V0):.5f} km/s") # starting altitude and speed

    # starting elements, angles printed in degrees
    print("\nINITIAL ORBITAL ELEMENTS (Algorithm 4.2)")
    print(f"  h      = {h:14.4f} km^2/s")
    print(f"  i      = {degrees(i):14.6f} deg")
    print(f"  Omega  = {degrees(Om):14.6f} deg")
    print(f"  e      = {e:14.8f}")
    print(f"  omega  = {degrees(om):14.6f} deg")
    print(f"  theta0 = {degrees(th0):14.6f} deg") # ~360 so it starts right at perigee
    print(f"  a      = {a:14.4f} km")
    print(f"  T      = {T:14.4f} s  ({T / 3600:.4f} h)")

    # solver settings so its all written down in the output
    print("\nINTEGRATION")
    print(f"  solver   : scipy solve_ivp, method RK45")
    print(f"  interval : 0 to {t_out[-1]:.2f} s ({t_out[-1] / 3600:.3f} h)"
          f" = {N_PERIODS:g} periods")
    print(f"  outputs  : {N_OUT} points, every {t_out[1]:.3f} s")
    print(f"  rtol = {RTOL_FINAL:g}, atol = {ATOL_FINAL:g}")

    # run the solver with the final tolerances!!
    sol = propagate(R0, V0, t_out, RTOL_FINAL, ATOL_FINAL)
    (t_max, alt_max, r_max), (t_grid, alt_grid) = first_altitude_max(sol) # the problem 2.4 answer

    # exact apogee to compare: r_a = a (1 + e), at the time where Me = pi
    alt_exact = a * (1.0 + e) - RE
    r_dummy, v_dummy, t_exact = kepler_state(R0, V0, 0.0) # only need t_apogee, r and v not used

    # answer 3 ways + the book answer
    print("\nPROBLEM 2.4 RESULT : first altitude maximum after t0")
    print(f"  from solver event (v_r = 0) : {alt_max:.3f} km at {t_max:.2f} s"
          f" = {t_max / 3600:.4f} h")
    print(f"  from output samples         : {alt_grid:.3f} km at {t_grid:.2f} s"
          f" = {t_grid / 3600:.4f} h")
    print(f"  exact two-body (Kepler)     : {alt_exact:.3f} km at {t_exact:.2f} s"
          f" = {t_exact / 3600:.4f} h")
    print(f"  Curtis answer (ode45)       : 9670 km at 1.66 h") # back of the book

    # check the whole RK45 path against my Task 3 Kepler method
    print("\nCHECK : RK45 position minus Kepler's-equation position (Task 3 method)")
    for k in [N_OUT // 8, N_OUT // 4, N_OUT // 2, N_OUT - 1]: # a few times along the way
        r_kep, v_kep, t_apo = kepler_state(R0, V0, sol.t[k])
        dr = np.linalg.norm(sol.y[0:3, k] - r_kep) # how far apart the 2 answers are, km
        print(f"  t = {sol.t[k]:9.2f} s ({sol.t[k] / T:.3f} T) : |dr| = {1000 * dr:.4f} m") # x1000 to get meters

    # how constant do the elements stay?
    elems = elements_history(sol)
    names = ["h (km^2/s)", "i (deg)", "Omega (deg)", "e", "omega (deg)"] # left out theta, its supposed to change
    print("\nLARGEST CHANGE OF EACH ELEMENT FROM ITS INITIAL VALUE")
    for col, name in enumerate(names):
        print(f"  {name:12s}: {np.max(np.abs(elems[:, col] - elems[0, col])):.3e}") # biggest drift from the start
    r_mag = np.linalg.norm(sol.y[0:3], axis=0)
    speed = np.linalg.norm(sol.y[3:6], axis=0)
    energy = speed**2 / 2.0 - MU / r_mag # specific energy, should stay constant
    print(f"  {'energy':12s}: {np.max(np.abs(energy - energy[0])):.3e} (km/s)^2")
    print(f"  altitude range : {r_mag.min() - RE:.2f} to {r_mag.max() - RE:.2f} km") # perigee to apogee
    print(f"  speed range    : {speed.min():.4f} to {speed.max():.4f} km/s")

    tolerance_study(t_out, alt_exact, t_exact) # runtime table

    # figures
    os.makedirs(FIG_DIR, exist_ok=True) # makes the folder if its not there yet
    figs = [("fig1_orbit_3d.png", plot_orbit_3d(sol, r_max)),
            ("fig2_time_histories.png", plot_time_histories(sol, t_max, alt_max)),
            ("fig3_orbital_elements.png", plot_elements(sol, elems))]
    print("\nFIGURES")
    for name, fig in figs: # save each one as a png
        path = os.path.join(FIG_DIR, name)
        fig.savefig(path, dpi=200) # dpi 200 so it looks sharp in the report
        print(f"  saved {path}")


if __name__ == "__main__": # only runs main if you run this file directly
    main()