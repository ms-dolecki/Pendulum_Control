from sympy import symbols, Eq, Or, cos, sin
from sympy.plotting import plot_implicit
from math import pi

def plot_equilibrium_equations(l):
    theta1, theta2 = symbols('theta1 theta2')
    g = 10
    eq1 = Eq(-g*cos(theta1)+l**2*((5/2)*cos(theta1)+sin(theta2-theta1)), 0)
    eq2 = Eq(-g*cos(theta2)-l**2*((5/2)*cos(theta2)-sin(theta1-theta2)), 0)
    combined_eq = Or(eq1, eq2)
    plot_implicit(combined_eq, (theta1, 0, 2*pi), (theta2, 0, 2*pi), title='l='+str(l))

for index in range(100):
    plot_equilibrium_equations(index*0.05)