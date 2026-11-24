"""
Koth_Amelia_Task3.py  -  EP393 Task 3

Find where a satellite is 50 minutes later using two-body motion.

References
  [1] Curtis, Orbital Mechanics for Engineering Students, 4th ed. (2021)
        Algorithm 3.1 (p. 148)
        Algorithm 4.5 (p. 210) orbital elements 
  [2] EP393 equation sheet for the individual formulas

Not used: universal variables, Lagrange f and g, scipy root finders.
"""
import sys
import io # to capture the printed output for the pdf
from math import sqrt, sin, cos, acos, atan2, pi, degrees
import numpy as np
import matplotlib # only used to write the output pdf
matplotlib.use("Agg") # no window, just save to file
import matplotlib.pyplot as plt


# Given values

MU = 398600.0 # Earth mu, km^3/s^2
R0 = np.array([6472.1, -7470.0, -2469.8]) # initial position, km
V0 = np.array([3.9914, 2.7916, -3.2948]) # initial velocity, km/s
DT = 3000.0 # 50 min in seconds
K_HAT = np.array([0.0, 0.0, 1.0]) # K unit vector (z axis)

TWO_PI = 2.0 * pi 


def sgn(x):
    """Returns +1 or -1 depending on the sign of x (0 counts as +)."""
    return -1.0 if x < 0.0 else 1.0 # negative -> -1, else +1


def wrap(angle):
    """Puts an angle in radians into the range 0 to 2*pi."""
    return angle % TWO_PI # mod 2pi handles negatives too


def safe_acos(x):
    """acos that doesn't crash if rounding gives something like 1.0000000002."""
    return acos(max(-1.0, min(1.0, x))) # clamp to [-1, 1] then acos


# 1. State vector -> orbital elements

def state_to_elements(r_vec, v_vec, mu):
    """Turn position and velocity into the classical orbital elements.

    In : r_vec (km), v_vec (km/s), mu (km^3/s^2)
    Out: h (km^2/s), e, a (km), i, Omega, omega, theta0 (all rad), T (s)
    """
    r = np.linalg.norm(r_vec) # distance from Earth center
    v = np.linalg.norm(v_vec) # speed

    # angular momentum h = r x v
    h_vec = np.cross(r_vec, v_vec) # h vector norm to the orbit
    h = np.linalg.norm(h_vec) # its magnitude

    # eccentricity vector, points at perigee
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r # e = (v x h)/mu - r/|r|
    e = np.linalg.norm(e_vec) # eccentricity

    # node vector N = K x h, points at the ascending node
    N_vec = np.cross(K_HAT, h_vec) # N vector
    N = np.linalg.norm(N_vec) # its magnitude
    if N < 1e-12: # N = 0 means i = 0, no node line
        sys.exit("ERROR: both omega's are undefined because the orbit is equatorial (i = 0)")

    # energy eps = v^2/2 - mu/r, and eps = -mu/(2a), so solve for a
    eps = v**2 / 2.0 - mu / r # specific energy
    a = -mu / (2.0 * eps) # semimajor axis

    # true anomaly radial velocity tells us if going toward or away
    # from perigee, which picks the sign
    v_r = np.dot(r_vec, v_vec) / r # radial speed component
    theta0 = sgn(v_r) * safe_acos((1.0 / e) * (h**2 / (mu * r) - 1.0)) # from orbit eq

    # inclination = angle between h and K
    i = safe_acos(np.dot(h_vec, K_HAT) / h) # cos i = h_z / h

    # argument of perigee = angle from N to e, sign from e_z
    omega = sgn(e_vec[2]) * safe_acos(np.dot(N_vec, e_vec) / (N * e)) # e_z<0 -> omega>180

    # RAAN = angle from I to N, sign from N_y
    Omega = sgn(N_vec[1]) * safe_acos(N_vec[0] / N) # N_y<0 -> Omega>180

    # period
    T = TWO_PI * sqrt(abs(a)**3 / mu) # T = 2pi sqrt(a^3/mu)

    return h, e, a, i, wrap(Omega), wrap(omega), wrap(theta0), T # angles 0..2pi


# 2. Kepler's equation with Newton's method (Curtis Algorithm 3.1, p. 148)

def kepler_newton(Me, e, tol=1e-8, max_iter=50):
    """Solve Me = E - e*sin(E) for E using Newton's method (Algorithm 3.1).

    In : Me (rad) mean anomaly, e eccentricity,
         tol (rad) stop when E changes by less than this,
         max_iter give up after this many loops
    Out: E (rad) eccentric anomaly, n_iter how many loops it took
    Prints an error and quits if it never converges.
    """
    # Newton: E_new = E - f/f'  where f = E - e sin E - Me, f' = 1 - e cos E
    # starting guess from the textbook
    E = Me + e / 2.0 if Me < pi else Me - e / 2.0 # first guess for E
    n_iter = 0 # loop counter
    while True: # loop until converged or give up
        n_iter += 1 # count this pass
        f = E - e * sin(E) - Me# Kepler's eq moved to one side
        fp = 1.0 - e * cos(E) # derivative of f wrt E
        dE = f / fp # Newton step
        E = E - dE # update E
        if abs(dE) < tol: # change small enough? done
            break
        if n_iter >= max_iter: # too many loops quit
            sys.exit(f"ERROR: Newton's method did not converge in!!!! "
                     f"{max_iter} iterations (last |dE| = {abs(dE):.3e} rad).")
    print(f"  Newton iterations      : {n_iter}") # show how many
    print(f"  Final E                : {E:.10f} rad  ({degrees(E):.6f} deg)") # show result
    return E, n_iter


# 3. Orbital elements -> state vector (Curtis Algorithm 4.5, p. 210)

def R1(ang):
    """Rotation matrix about the x axis."""
    c, s = cos(ang), sin(ang) # shorthand
    return np.array([[1.0, 0.0, 0.0], # x stays the same
                     [0.0,   c,   s], # y, z rotate
                     [0.0,  -s,   c]])


def R3(ang):
    """Rotation matrix about the z axis."""
    c, s = cos(ang), sin(ang) # shorthand
    return np.array([[  c,   s, 0.0], # x, y rotate
                     [ -s,   c, 0.0],
                     [0.0, 0.0, 1.0]]) # z stays the same


def elements_to_state(h, e, i, Omega, omega, theta, mu):
    """Orbital elements -> r and v in the geocentric equatorial frame
    (Algorithm 4.5).

    In : h (km^2/s), e, i, Omega, omega, theta (rad), mu (km^3/s^2)
    Out: r_vec (km), v_vec (km/s)
    """
    # orbit equation gives the distance
    r = (h**2 / mu) / (1.0 + e * cos(theta)) # r = (h^2/mu)/(1+e cos th)

    # r and v in the perifocal frame (x toward perigee, z along h)
    r_pqw = r * np.array([cos(theta), sin(theta), 0.0]) # position in perifocal
    v_pqw = (mu / h) * np.array([-sin(theta), e + cos(theta), 0.0]) # velocity in perifocal

    # rotation from geocentric to perifocal is R3(omega) R1(i) R3(Omega),
    # so going the other way is the transpose
    C = R3(omega) @ R1(i) @ R3(Omega) # geocentric -> perifocal
    r_vec = C.T @ r_pqw # transpose goes perifocal -> geocentric
    v_vec = C.T @ v_pqw # same for velocity
    return r_vec, v_vec



# main

def line(title):
    print(f"\n{title}") # section header


def main():
    print("EP393 Task 3 : two-body propagation, dt = 3000 s (50 min)")

    # step 1: orbital elements at t0
    h, e, a, i, Om, om, th0, T = state_to_elements(R0, V0, MU) # unpack all 8

    line("INITIAL STATE (t0)")
    print(f"  r0 = [{R0[0]:12.4f} {R0[1]:12.4f} {R0[2]:12.4f}] km")
    print(f"  v0 = [{V0[0]:12.6f} {V0[1]:12.6f} {V0[2]:12.6f}] km/s")

    line("INITIAL ORBITAL ELEMENTS")
    print(f"  h      = {h:14.4f} km^2/s")
    print(f"  e      = {e:14.8f}")
    print(f"  a      = {a:14.4f} km")
    print(f"  i      = {degrees(i):14.6f} deg") # print angles in degrees
    print(f"  Omega  = {degrees(Om):14.6f} deg")
    print(f"  omega  = {degrees(om):14.6f} deg")
    print(f"  theta0 = {degrees(th0):14.6f} deg")
    print(f"  T      = {T:14.4f} s  ({T/60:.3f} min)")

    # Kepler's equation only works for ellipses, so make sure e < 1
    if not (0.0 <= e < 1.0): # parabola or hyperbola?
        sys.exit(f"ERROR: e = {e:.6f} is not < 1, orbit is not an ellipse.")
    print("  Check: e < 1  ->  elliptical orbit.")

    # step 2: get the mean anomaly at t0, then move it forward by dt
    line("TIME OF FLIGHT")
    # true anomaly -> eccentric anomaly. using atan2 with sin and cos
    # instead of tan so the quadrant comes out right
    E0 = 2.0 * atan2(sqrt(1.0 - e) * sin(th0 / 2.0), # tan(E/2) = sqrt((1-e)/(1+e)) tan(th/2)
                     sqrt(1.0 + e) * cos(th0 / 2.0))
    E0 = wrap(E0) # keep in 0..2pi

    # Kepler's equation forward: Me = E - e sin E
    Me0 = wrap(E0 - e * sin(E0)) # mean anomaly at t0

    # mean anomaly grows at a constant rate n = 2*pi/T
    Me = wrap(Me0 + (TWO_PI / T) * DT) # mean anomaly after dt

    print(f"  E0     = {degrees(E0):14.6f} deg")
    print(f"  Me0    = {degrees(Me0):14.6f} deg  ({Me0:.10f} rad)")
    print(f"  Me     = {degrees(Me):14.6f} deg  ({Me:.10f} rad)  [after dt]")

    # step 3: solve Kepler's equation for the new E
    line("KEPLER'S EQUATION (Newton, Algorithm 3.1)")
    E, n_iter = kepler_newton(Me, e, tol=1e-8, max_iter=50) # Algorithm 3.1

    # step 4: new E -> new true anomaly (same atan2 trick, reversed)
    theta = wrap(2.0 * atan2(sqrt(1.0 + e) * sin(E / 2.0), # tan(th/2) = sqrt((1+e)/(1-e)) tan(E/2)
                             sqrt(1.0 - e) * cos(E / 2.0)))
    line("FINAL TRUE ANOMALY")
    print(f"  theta  = {degrees(theta):14.6f} deg")

    # step 5: elements with the new theta -> final r and v
    r_f, v_f = elements_to_state(h, e, i, Om, om, theta, MU) # Algorithm 4.5

    line("FINAL STATE (t0 + 3000 s)")
    print(f"  r = [{r_f[0]:12.4f} {r_f[1]:12.4f} {r_f[2]:12.4f}] km")
    print(f"  |r| = {np.linalg.norm(r_f):12.4f} km")
    print(f"  v = [{v_f[0]:12.6f} {v_f[1]:12.6f} {v_f[2]:12.6f}] km/s")
    print(f"  |v| = {np.linalg.norm(v_f):12.6f} km/s")


def save_pdf(text, filename="Koth_Amelia_Task3.pdf"):
    """Write the console output text to a one page pdf.

    In : text (str) everything that was printed, filename (str)
    Out: nothing, just saves the file
    """
    fig = plt.figure(figsize=(8.5, 11)) # letter size page
    fig.text(0.06, 0.97, "Amelia Koth - EP393 Task 3 - console output of Koth_Amelia_Task3.py",
             fontsize=11, weight="bold", va="top") # title
    fig.text(0.06, 0.95, "Algorithms 3.1 (p. 148) and 4.5 (p. 210) from Curtis, "
             "Orbital Mechanics for Engineering Students, 4th ed. (2021)",
             fontsize=8, va="top") # citation line
    fig.text(0.06, 0.93, text, family="monospace", fontsize=7.6, va="top") # the output
    fig.savefig(filename) # save it
    plt.close(fig)


if __name__ == "__main__":
    buf = io.StringIO() # holds a copy of everything printed
    real_stdout = sys.stdout # remember the real console
    sys.stdout = buf # send prints into the buffer
    try:
        main() # run everything
    finally:
        sys.stdout = real_stdout # always put the console back
    output = buf.getvalue()
    print(output, end="") # show it in the terminal like normal
    save_pdf(output) # and also save it as a pdf
    print(f"\nSaved Koth_Amelia_Task3.pdf")
