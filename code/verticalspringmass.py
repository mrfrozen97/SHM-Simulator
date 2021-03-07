import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import warnings
warnings.filterwarnings("ignore")








class Horizontal_spring_mass():

    def __init__(self, mass=0.5, spring_constant=2.5, amplitude=10, damping_coefficient=0.9, gravity=9.8, display_equilibrium=True):
        self.window_sizeX = 7
        self.window_sizeY = 7

        self.display_equilibrium = display_equilibrium
        self.gravity = gravity
        self.amplitude = amplitude
        self.mass = mass
        self.spring_constant = spring_constant
        self.angular_frequency = math.sqrt(self.spring_constant/self.mass)
        self.total_energy =[]
        self.equilibrium_position = (self.mass*self.gravity)/self.spring_constant



        self.phi = 0
        self.plot_x = np.array([i for i in range(400)])
        self.plot_y_position = []
        self.plot_y_velocity = []
        self.plot_y_accelaration = []
        self.time_x = []
        self.damping_ratio = damping_coefficient
        self.damped_angular_frequency = self.angular_frequency * math.sqrt(1-self.damping_ratio**2)
        self.plot_y_velocity_total = []
        self.plot_y_accelaration_total = []
        self.plot_y_position_total = []
        self.time_x_total = []

        temp_vel = self.amplitude * self.angular_frequency
        max_value = max(max(self.angular_frequency**2 * self.amplitude,temp_vel), self.amplitude)

        self.y_ticks_array = [i for i in range(int(-max_value*1.2), int(max_value*1.2), int(max_value/4))]

        #self.x_spring = np.array([i / 40 for i in range(-400, 635)])

        if self.amplitude>=8:
            self.x_spring = np.array([i/10 for i in range(int(-self.amplitude*1.8)*10, int(self.amplitude*1.5)*10)])
        elif self.amplitude>=1:
            self.x_spring = np.arange(-self.amplitude*1.8, self.amplitude*1.8, 0.01)
        else:
            self.x_spring = np.arange(-2, self.amplitude * 1.8, 0.04)

        self.spring_y_wall_position = min(-self.amplitude*1.5, -2)
        self.spring_figure_xticks_count = 3
        self.x_ticks_array = [-1, -2, 0, 1, 2]



    def calculate(self, i):

        plt.clf()

        i = i/10

        self.time_x.append(i)
        self.time_x_total.append(i)
        self.angular_frequency = math.sqrt(self.spring_constant/self.mass)

        self.y = self.amplitude *math.exp(-1*self.damping_ratio*self.angular_frequency*i) * math.cos(self.damped_angular_frequency*i + self.phi)
        #self.y = self.y - self.equlibrim_position
        temp_y_dif = self.y-self.equilibrium_position
        self.plot_y_position.append(temp_y_dif)
        temp_vel = self.amplitude * self.angular_frequency*math.exp(-1*self.damping_ratio*self.angular_frequency*i)* math.sin(self.damped_angular_frequency*i + self.phi)
        self.plot_y_velocity.append(temp_vel)
        self.plot_y_velocity_total.append(temp_vel)
        self.plot_y_position_total.append(self.y)



        #self.plot_y_accelaration.append(self.amplitude *-1*self.damped_angular_frequency**2* math.cos(self.damped_angular_frequency*i + self.phi))
        temp_accn = -1*self.angular_frequency**2 * self.y
        self.plot_y_accelaration.append(temp_accn)
        self.plot_y_accelaration_total.append(temp_accn)
        if len(self.time_x)>=100:
            self.time_x = self.time_x[-100:]
            self.plot_y_position = self.plot_y_position[-100:]
            self.plot_y_velocity = self.plot_y_velocity[-100:]
            self.plot_y_accelaration = self.plot_y_accelaration[-100:]



        distance = self.y - self.equilibrium_position
        dummy_amplitude = self.amplitude + self.equilibrium_position

        #self.y_spring = np.array([math.sin(((index * 150 * math.pi) / 180) * ((self.amplitude-distance/2)/(self.amplitude*0.8)))*2 + 5 for index in self.x_spring])/ 5

        if self.amplitude>=1:
            self.y_spring = np.array([math.sin((((index/(self.amplitude) *350* math.pi) / 18)) * (dummy_amplitude + distance*0.8)/(dummy_amplitude)) * 0.8 + 5 for index in self.x_spring]) / 5
        else:
            self.y_spring = np.array([math.sin(
                ((index * 80 * math.pi) / 18) * (self.amplitude - distance) / (
                    self.amplitude)) * 2 + 5 for index in self.x_spring]) / 5

        temp_x = [x for x in self.x_spring if x > distance]
        temp_y = self.y_spring[:len(temp_x)]


        plt.figure(1)


        plt.subplot(211)
        #plt.plot(self.time_x, self.plot_y_position)
        plt.title("SHM Motion and plot")
        plt.plot(self.time_x, self.plot_y_velocity, label="Velocity")
        plt.plot(self.time_x, self.plot_y_accelaration, label="Accelaration")
        plt.xlabel("Time (in seconds)")
        plt.ylabel("Amplitude")
        plt.yticks(self.y_ticks_array)
        plt.legend(loc="upper left", fontsize='small')




        plt.subplot(212)
        plt.plot(temp_y, temp_x, zorder=1)
        plt.scatter([1], [distance], color='r', s=1100, zorder=2, alpha=1)
        #plt.plot([-1*self.amplitude*1.2, 16-self.amplitude, 16-self.amplitude], [0.2, 0.2, 4])

        if self.display_equilibrium:
            plt.plot([-2, 4], [-self.equilibrium_position, -self.equilibrium_position], color='b', linestyle='--', linewidth=0.8)

        plt.plot([-2, 4], [-self.spring_y_wall_position,-self.spring_y_wall_position])
        #x_ticks_array = [2 * i for i in range(-1*int(self.amplitude/1.4),int(self.amplitude), int(self.amplitude/self.spring_figure_xticks_count))]


        plt.yticks(self.x_ticks_array)
        plt.xticks([2*i for i in range(-1, 3)])











    def animate(self):
        plt.style.use('dark_background')
        fig = plt.figure()

        #x_ticks_array = [i for i in range(int(-self.amplitude * 1.5), int(self.amplitude * 1.5),max(int(self.amplitude / self.spring_figure_xticks_count), 1))]
        x_ticks_array = []

        for value in range(0, int(self.amplitude*1.5 + self.equilibrium_position), max(int((self.amplitude+self.equilibrium_position) / self.spring_figure_xticks_count), 1)):
            x_ticks_array.append(value)
            x_ticks_array.append(-value)

        if len(x_ticks_array)>=len(self.x_ticks_array):
            self.x_ticks_array = x_ticks_array



        fig.set_size_inches(self.window_sizeX, self.window_sizeY)
        ani = FuncAnimation(fig, self.calculate, interval=50, frames=10000)


        plt.show()


    def plot_all_points(self):

        initial_energy = self.mass*self.gravity*(self.amplitude) + (self.spring_constant*(self.equilibrium_position)**2)/2

        for i in range(len(self.plot_y_velocity_total)):
            temp_kinetic = self.mass*(self.plot_y_velocity_total[i]**2)/2

            temp_potential = (self.spring_constant/2) * (abs(self.plot_y_position_total[i]))**2

            self.total_energy.append(temp_kinetic+ temp_potential + initial_energy)
            self.plot_y_position_total[i] -= self.equilibrium_position




        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.set_size_inches(int(self.window_sizeX*1.6), self.window_sizeY)
        fig.suptitle("Simple Harmonic Motion")
        ax1.plot(self.time_x_total, self.plot_y_accelaration_total, label="Accelaration", color='#FFA500')
        ax2.plot(self.time_x_total, self.plot_y_velocity_total, label="Velocity", color='#EE82EE')
        ax3.plot(self.time_x_total, self.plot_y_position_total, label="Position", color='#00FFFF')
        ax4.plot(self.time_x_total, self.total_energy, label="Total energy", color='#FFFF00')

        ax1.set_title("Acceleration  vs  Time (in seconds)")
        ax2.set_title("Velocity  vs  Time (in seconds)")
        ax3.set_title("Position vs  Time (in seconds)")
        ax4.set_title("Total Energy vs  Time (in seconds)")
        fig.tight_layout()
        plt.show()




plt2 = Horizontal_spring_mass(mass=0.5, spring_constant=2.5,
                              amplitude=10, damping_coefficient=0.03, gravity=9.8, display_equilibrium=False)

plt2.animate()
plt2.plot_all_points()