import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt




##############################################################################################################################3

# 1D SHM


"""
m = 10
k = 2.5
g = 9.8
c = 0.02

def model(x, t):

    x1 = x[0]
    x2 = x[1]

    dx1dt = x2
    dx2dt = ((m*g)-(c*x2)-(k*x1))

    return [dx1dt, dx2dt]

t = np.linspace(0, 300, 1000)

pos = odeint(model,[0, 0], t)

x = []
v = []

for i in pos:
    x.append(i[0])
    v.append(i[1])

print(x, v)

plt.plot(t,x)
plt.plot(t,v)
plt.show()
"""





"""
##############################################################################################################################3

# 2D SHM


m = 10
k = 6
g = 9.8
c = 0.02
R = 5
initial_position = [5, 25]
initial_velocity = [4, 10]


def model(x, t):

    x1 = x[0]
    x2 = x[1]
    y1 = x[2]
    y2 = x[3]
    L = math.sqrt(x1**2 + y1**2)
    S = L-R
    sin0 = x1/L
    cos0 = y1/L
    dx1dt = x2
    dx2dt = (((-k/m)*S*sin0) - (c*x2))

    dy1dt = y2
    dy2dt = (((-k / m)*S*cos0) - (c * y2))


    return [dx1dt, dx2dt, dy1dt, dy2dt]

t = np.linspace(0, 300, 1000)

pos = odeint(model,[initial_position[0], initial_velocity[0], initial_position[1], initial_velocity[1]], t)

x = []
vx = []
y = []
vy = []
v = []

for i in pos:
    x.append(i[0])
    vx.append(i[1])
    y.append(i[2])
    vy.append(i[3])

plt.plot(t,x)
plt.plot(t,vx)
plt.plot(t, y)
plt.plot(t, vy)
plt.show()

"""






#2D SHM OOP(object oriented programming) implementation

class SHM2D:

    def __init__(self):
        self.m = 10                                  # Mass of the block
        self.k = 2.5                                 # Spring constant
        self.g = 9.8
        self.c = 0.02                                # Damping coefficient
        self.R = 5                                   # Initial length of the spring
        self.initial_position = [5, 50]              # Initial position of the block
        self.initial_velocity = [3, 8]               # Initial velocity of the block
        self.t = np.linspace(0, 300, 1000)           # Time with 1000 unique points
        self.pos = []                                # Stores values to plot






    # Method to implement the function for the differential equation.....
    def model(self, x, t):
        x1 = x[0]
        x2 = x[1]
        y1 = x[2]
        y2 = x[3]
        L = math.sqrt(x1 ** 2 + y1 ** 2)
        S = L - self.R
        sin0 = x1 / L
        cos0 = y1 / L


        # For x-axis
        dx1dt = x2
        dx2dt = (((-self.k / self.m) * S * sin0) - (self.c * x2))

        # For y-axis
        dy1dt = y2
        dy2dt = (((-self.k / self.m) * S * cos0) - (self.c * y2))

        return [dx1dt, dx2dt, dy1dt, dy2dt]






    # Method to calculate the values and return it.................
    def get_values(self):

        pos = odeint(self.model, [self.initial_position[0], self.initial_velocity[0], self.initial_position[1], self.initial_velocity[1]], self.t)
        self.pos = pos
        #return pos





    # Method to plot curve for the calcualted values..................
    def plot_curve(self):
        x = []
        vx = []
        y = []
        vy = []

        for i in self.pos:
            x.append(i[0])
            vx.append(i[1])
            y.append(i[2])
            vy.append(i[3])




        plt.style.use('dark_background')
        fig = plt.figure()
        fig.set_size_inches(8, 6)

        plt.figure(1)
        plt.subplot(211)
        plt.plot(self.t, x, label="displacement x-axis")
        plt.plot(self.t, y, label="displacement y-axis")
        plt.legend()

        plt.subplot(212)
        plt.plot(self.t, vx, label="velocity x-axis")
        plt.plot(self.t, vy, label="velocity y-axis")
        plt.legend()
        plt.xlabel("Time (in seconds)")
        plt.show()



sim = SHM2D()
sim.get_values()
sim.plot_curve()



