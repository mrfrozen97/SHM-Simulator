import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Spring_Pendulum:

    def __init__(self):
        self.mass = 2.5
        self.spring_constant = 12.253
        self.gravity = 9.81
        self.initial_length = 60
        self.dt = 0.01
        self.time = np.arange(0.0, 200.0, 0.01)

        self.displacement = []
        self.tita = []
        self.velocity = []
        self.ux = []
        self.uy = []
        self.input = np.array([0.0, 0.0, 30.0, 0.28])
        self.animation_xticks = [-2, -1, 0, 1, 2]
        self.animaton_ticks_count = 2
        self.animation_yticks = [-2, -1, 0, 1, 2]
        self.springx = []
        self.springy = []



    def calculate_dd(self, input, t):

        xd, titad, x, tita = input[0], input[1], input[2], input[3]

        xdd = (self.initial_length+x) * titad**2 - (self.spring_constant*x)/self.mass + self.gravity*math.cos(tita)
        titadd = (-2*xd*titad)/(self.initial_length + x) - self.gravity*math.sin(tita)/(self.initial_length+x)

        return np.array([xdd, titadd, xd, titad])

    def runga_kutta(self, input, t):

        k1 = self.calculate_dd(input, t)
        k2 = self.calculate_dd(input + (k1*self.dt)*0.5, t+self.dt*0.5)
        k3 = self.calculate_dd(input + (k2 *self.dt)*0.5, t + self.dt*0.5)
        k4 = self.calculate_dd(input + (k3*self.dt), t+self.dt)


        return (self.dt*(k1 + 2*k2 + 2*k3 + k4)/6)





    def calculate(self):
        for t in self.time:
            self.input = self.input + obj.runga_kutta(self.input, t)
            self.displacement.append(self.input[2])
            self.velocity.append(self.input[0])
            self.tita.append(self.input[3])
            self.ux.append(-self.input[2] * math.sin(self.input[3]))
            self.uy.append(-self.input[2] * math.cos(self.input[3]))









    def animate_calculations(self, i):



        distance = abs(self.displacement[5*i])
        ratio = max(0.05, distance/max(self.displacement))
        multiplier_sin = 0.2*max(self.animation_yticks)/4.35
        mul_in = 8*4.35/max(self.animation_yticks)
        self.springx = [multiplier_sin*math.sin(y/ratio*mul_in) for y in self.springy if y>=-distance]


        sign_dir = 1
        if self.uy[5*i]>0:
            sign_dir=-1

        temp_tita = [math.cos(self.tita[5*i]), math.sin(self.tita[5*i])]
        xnew = [sign_dir*(self.springx[ind]*temp_tita[0] + self.springy[ind]*temp_tita[1]) for ind in range(len(self.springx))]
        ynew = [sign_dir*(self.springy[ind]*temp_tita[0] - self.springx[ind]*temp_tita[1]) for ind in range(len(self.springx))]



        #print(distance)


        plt.clf()

        plt.figure(1)
        plt.subplot(211)

        min = max(0, i*5 - 500)
        plt.plot(self.ux[:i*5], self.uy[:i*5], linewidth=0.5, color='g')




        plt.subplot(212)
        #plt.plot([0, self.ux[5 * i]], [0, self.uy[5 * i]], zorder=1)
        plt.scatter(self.ux[5 * i], self.uy[5 * i], s=500, color='r', zorder=3)
        plt.scatter([0], [0], s=50, zorder=2, color='y')
        #plt.plot(self.springx, self.springy)
        plt.plot(xnew, ynew, zorder=1)
        plt.xticks(self.animation_xticks)
        plt.yticks(self.animation_yticks)



    def animate(self):

        max_vx = max(abs(max(self.ux)), abs(min(self.ux)))
        max_vy = max(abs(max(self.uy)), abs(min(self.uy)))
        self.springy = np.arange(0, max(self.displacement)*1.1, max(self.displacement)/100)
        self.springy = self.springy*-1


        self.animation_xticks = np.arange(-1.2 * max_vx, 1.5 * max_vx, max_vx / self.animaton_ticks_count)
        self.animation_yticks = np.arange(-1.2 * max_vy, 1.5 * max_vy, max_vy/ self.animaton_ticks_count)




        if max(self.animation_yticks) > max(self.animation_xticks) or abs(min(self.animation_yticks)) > abs(min(self.animation_xticks)):
            self.animation_xticks = self.animation_yticks
        else:
            self.animation_yticks = self.animation_xticks

        print(max(self.animation_yticks))

        plt.style.use('dark_background')
        fig = plt.figure()
        fig.set_size_inches(6, 8)
        ani = FuncAnimation(fig, self.animate_calculations, interval=50, frames=4000)

        plt.show()

        plt.plot(obj.ux, obj.uy, linewidth=0.5, color='g')
        # plt.plot(time, displacement)
        # plt.plot(time, velocity)
        # plt.plot(time, tita)
        plt.show()


obj = Spring_Pendulum()

obj.calculate()
obj.animate()



