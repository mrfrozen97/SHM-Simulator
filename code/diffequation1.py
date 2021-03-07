import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
import math
import warnings
warnings.filterwarnings("ignore")


#A generalised class to implement the simple harmonic motion





#Class that implements horizontal simple harmonic motion.

class Horizontal_spring_mass():

    def __init__(self, mass=0.5, spring_constant=2.5, amplitude=10, damping_coefficient=0.02):
        self.window_sizeX = 7
        self.window_sizeY = 7
        self.equilibrium_position = 0

        self.amplitude = amplitude
        self.mass = mass
        self.spring_constant = spring_constant
        self.angular_frequency = math.sqrt(self.spring_constant/self.mass)
        self.total_energy =[]


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

        self.y_ticks_array = [i for i in range(int(-max_value*1.2), int(max_value*1.2), max(int(max_value/4), 1))]

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
        self.plot_y_position.append(self.y)
        temp_vel = self.amplitude *self.angular_frequency*math.exp(-1*self.damping_ratio*self.angular_frequency*i)* math.sin(self.damped_angular_frequency*i + self.phi)
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



        distance = self.y

        #self.y_spring = np.array([math.sin(((index * 150 * math.pi) / 180) * ((self.amplitude-distance/2)/(self.amplitude*0.8)))*2 + 5 for index in self.x_spring])/ 5

        if self.amplitude>=1:
            self.y_spring = np.array([math.sin((((index/(self.amplitude) *350* math.pi) / 18)) * (self.amplitude - distance*0.8)/(self.amplitude)) * 2 + 5 for index in self.x_spring]) / 5
        else:
            self.y_spring = np.array([math.sin(
                ((index * 80 * math.pi) / 18) * (self.amplitude - distance) / (
                    self.amplitude)) * 2 + 5 for index in self.x_spring]) / 5

        temp_x = [x for x in self.x_spring if x < distance and x>self.spring_y_wall_position]
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
        plt.plot(temp_x, temp_y, zorder=1)
        plt.scatter([distance], [1], color='r', s=1100, zorder=2, alpha=1)
        #plt.plot([-1*self.amplitude*1.2, 16-self.amplitude, 16-self.amplitude], [0.2, 0.2, 4])

        plt.plot([self.spring_y_wall_position ,self.spring_y_wall_position, max(self.amplitude, 1.5)], [4, 0.15, 0.15])
        #x_ticks_array = [2 * i for i in range(-1*int(self.amplitude/1.4),int(self.amplitude), int(self.amplitude/self.spring_figure_xticks_count))]


        plt.xticks(self.x_ticks_array)
        plt.yticks([2*i for i in range(-1, 4)])











    def animate(self):
        plt.style.use('dark_background')
        fig = plt.figure()

        #x_ticks_array = [i for i in range(int(-self.amplitude * 1.5), int(self.amplitude * 1.5),max(int(self.amplitude / self.spring_figure_xticks_count), 1))]
        x_ticks_array = []

        for value in range(0, int(self.amplitude*1.5), max(int(self.amplitude / self.spring_figure_xticks_count), 1)):
            x_ticks_array.append(value)
            x_ticks_array.append(-value)

        if len(x_ticks_array)>=len(self.x_ticks_array):
            self.x_ticks_array = x_ticks_array



        fig.set_size_inches(self.window_sizeX, self.window_sizeY)
        ani = FuncAnimation(fig, self.calculate, interval=50, frames=10000)


        plt.show()


    def plot_all_points(self):


        for i in range(len(self.plot_y_velocity_total)):
            temp_kinetic = self.mass*(self.plot_y_velocity_total[i]**2)/2

            temp_potential = (self.spring_constant/2) * abs(self.plot_y_position_total[i])**2

            self.total_energy.append(temp_kinetic+ temp_potential)




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

















class Vertical_spring_mass():

    def __init__(self, mass=0.5, spring_constant=2.5, amplitude=10, damping_coefficient=0.9, gravity=9.8, display_equilibrium=True):
        self.window_sizeX = 7
        self.window_sizeY = 7

        self.display_equilibrium = display_equilibrium
        self.gravity = gravity
        self.amplitude = max(amplitude, 1)
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

        #max_value = max(max_value, 1)

        self.y_ticks_array = [i for i in range(int(-max_value*1.2), int(max_value*1.2), max(int(max_value/4), 1))]

        #self.x_spring = np.array([i / 40 for i in range(-400, 635)])

        dummy_amplitude = self.amplitude + self.equilibrium_position

        if self.amplitude>=8:
            self.x_spring = np.array([i/10 for i in range(int(-dummy_amplitude*1.8)*10, int(dummy_amplitude*1.8)*10)])
        elif self.amplitude>=1:
            self.x_spring = np.arange(-dummy_amplitude*1.8,dummy_amplitude*1.8, 0.01)
        else:
            self.x_spring = np.arange(-2, dummy_amplitude * 1.8, 0.04)

        self.spring_y_wall_position = min(-dummy_amplitude*1.5, -2)
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
            self.y_spring = np.array([math.sin((((index/(dummy_amplitude) *350* math.pi) / 18)) * (dummy_amplitude + distance*0.8)/(dummy_amplitude)) * 0.8 + 5 for index in self.x_spring]) / 5
        else:
            self.y_spring = np.array([math.sin(
                ((index * 80 * math.pi) / 18) * (self.amplitude - distance) / (
                    self.amplitude)) * 2 + 5 for index in self.x_spring]) / 5

        temp_x = [x for x in self.x_spring if x > distance and x<-self.spring_y_wall_position]
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

        dummy_amplitude = self.amplitude + self.equilibrium_position
        print(dummy_amplitude)
        temp_factor = 1.5
        if dummy_amplitude <= 4:
            temp_factor = 2

        for value in range(0, int(dummy_amplitude*temp_factor), max(int((dummy_amplitude) / self.spring_figure_xticks_count), 1)):
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









#Linear regression class ...................
class Linear_regression:


    def __init__(self):
        self.a = 0.0
        self.b = 0.0
        self.squared_mean_error = 0.0
        self.x_mean = 0
        self.y_mean = 0
        self.size = 1
        self.x = []
        self.y = []
        self.calculated_y = []
        self.calculated_x = []


    #Function to calcute and return a, b values....................................

    def bestFit(self, x_coordinates, y_coordinates):

        size_coordinates = np.size(x_coordinates)
        x_coordinates_mean = np.sum(x_coordinates)/size_coordinates
        y_coordinates_mean = np.sum(y_coordinates)/size_coordinates

        self.x = x_coordinates
        self.y = y_coordinates
        self.size = size_coordinates
        self.x_mean = x_coordinates_mean
        self.y_mean = y_coordinates_mean

        bestFint_numerator = (np.sum(np.multiply(x_coordinates, y_coordinates))/size_coordinates) - (x_coordinates_mean*y_coordinates_mean)
        bestFint_denominator = (np.sum(np.multiply(x_coordinates,x_coordinates))/size_coordinates) - (x_coordinates_mean**2)
        self.b = bestFint_numerator/bestFint_denominator
        self.a = y_coordinates_mean - (self.b*x_coordinates_mean)
        return self.a, self.b




    def bestFit_logrithemic(self, x_coordinates, y_coordinates, base=math.e):

        y_coordinates = [math.log(x, base) for x in x_coordinates]

        size_coordinates = np.size(x_coordinates)
        x_coordinates_mean = np.sum(x_coordinates)/size_coordinates
        y_coordinates_mean = np.sum(y_coordinates)/size_coordinates

        self.x = x_coordinates
        self.y = y_coordinates
        self.size = size_coordinates
        self.x_mean = x_coordinates_mean
        self.y_mean = y_coordinates_mean

        bestFint_numerator = (np.sum(np.multiply(x_coordinates, y_coordinates))/size_coordinates) - (x_coordinates_mean*y_coordinates_mean)
        bestFint_denominator = (np.sum(np.multiply(x_coordinates,x_coordinates))/size_coordinates) - (x_coordinates_mean**2)
        self.b = bestFint_numerator/bestFint_denominator
        self.a = y_coordinates_mean - (self.b*x_coordinates_mean)
        return self.a, self.b








    #function that calculates and returns squarred error which can be attributed to the accuracy og our best fit line............

    def squared_error(self):

        mean_y_array = [self.y_mean for x in range(self.size)]
        bestFit_y = [(self.b*x + self.a) for x in self.x]

        squared_error_denominator = np.sum(np.square(np.subtract(bestFit_y, mean_y_array)))
        squared_error_numerator = np.sum(np.square(np.subtract(bestFit_y, self.y)))


        self.squared_mean_error = 1 - (squared_error_numerator/squared_error_denominator)

        return self.squared_mean_error

    def find_unknowns(self, x):
        x = list(x)
        self.calculated_x = x
        for i in x:
            self.calculated_y.append(self.b * i + self.a)
        return self.calculated_y


















#  Class that implements envolope functions.........................................

class envolope:

    def __init__(self):

        self.displacement = []
        self.peaks = []
        self.peak_time = []
        self.equvilibrium = 0
        self.minimum_peak = 0


    def find_peaks(self, displacement, time, equvilibrium=0, minimum_peak=0):

        self.displacement = displacement
        self.equvilibrium = equvilibrium
        self.minimum_peak = minimum_peak

        temp_max = 0                                                 # Holds value for temp maximum
        temp_time = 0


        for i in range(1,len(self.displacement)):                  # Loops through all elements of the displacements

            if temp_max < abs(displacement[i]-self.equvilibrium):                     # If element greater than curret maximum. store it
                temp_max = abs(displacement[i]-self.equvilibrium)
                temp_time = time[i]





            if ((self.displacement[i-1] >= self.equvilibrium and self.displacement[i]<=self.equvilibrium) or (self.displacement[i-1]<=equvilibrium and self.displacement[i]>=equvilibrium)
                ) and abs(temp_max)>=self.minimum_peak:


                self.peaks.append(temp_max + equvilibrium)
                self.peak_time.append(temp_time)
                temp_max = 0

        return self.peaks, self.peak_time

























class GUI():

    def __init__(self):
        self.mass = 0.5
        self.spring_constant = 2.5
        self.amplitude = 10
        self.damping_coefficient = 0.02
        self.option = 1
        self.gravity = 9.8
        self.show_equilibrium_line = False

    def show_home(self):

        plt.style.use('grayscale')

        axbutton = plt.axes([0.3, 0.5, 0.4, 0.075])
        axbutton1 = plt.axes([0.3, 0.3, 0.4, 0.075])
        title1 = plt.axes([0.5, 0.8, 0.0, 0.0])

        title_box = Button(title1, "SHM Simulator")
        title_box.label.set_fontsize(24)

        startButton = Button(axbutton, "Horizontal SHM")
        startButton1 = Button(axbutton1, "Vertical SHM")

        startButton.on_clicked(self.set_option_horizontal)
        startButton1.on_clicked(self.set_option_verticle)
        title_box.on_clicked(self.do_nothing)


        plt.show()



    def set_option_verticle(self, texgt):
        self.option = 2
        plt.close()


    def set_option_horizontal(self, text):
        self.option = 1
        plt.close()




    def show_gui(self):

        plt.style.use('grayscale')

        axbox = plt.axes([0.3, 0.6, 0.4, 0.075])
        axbox1 = plt.axes([0.3, 0.5, 0.4, 0.075])
        axbox2 = plt.axes([0.3, 0.4, 0.4, 0.075])
        title1 = plt.axes([0.5, 0.8, 0.0, 0.0])
        axbox3 = plt.axes([0.3, 0.3, 0.4, 0.075])
        axbutton = plt.axes([0.5, 0.1, 0.4, 0.075])

        title_box = Button(title1, "SHM Horizontal Simulator")
        title_box.label.set_fontsize(24)

        text_box = TextBox(axbox, 'Mass(Kg)', initial=0.5)
        text_box1 = TextBox(axbox1, 'Spring constant(N/m)', initial=2.5)
        text_box2 = TextBox(axbox2, 'Amplitude(m)', initial=10)
        text_box3 = TextBox(axbox3, 'Damping Coefficient', initial=0.02)
        startButton = Button(axbutton, "START")

        startButton.on_clicked(self.button_pressed)
        title_box.on_clicked(self.do_nothing)
        text_box.on_submit(self.set_mass)
        text_box1.on_submit(self.set_spring_constant)
        text_box2.on_submit(self.set_amplitude)
        text_box3.on_submit(self.set_damping)




        plt.show()


    def show_gui1(self):

        plt.style.use('grayscale')

        axbox = plt.axes([0.3, 0.65, 0.4, 0.075])
        axbox1 = plt.axes([0.3, 0.55, 0.4, 0.075])
        axbox2 = plt.axes([0.3, 0.45, 0.4, 0.075])
        title1 = plt.axes([0.5, 0.85, 0.0, 0.0])
        axbox3 = plt.axes([0.3, 0.35, 0.4, 0.075])
        axbox4 = plt.axes([0.3, 0.25, 0.4, 0.075])
        axbutton = plt.axes([0.5, 0.1, 0.4, 0.075])
        check_box = plt.axes([0.1, 0.04, 0.35, 0.2])

        title_box = Button(title1, "SHM Vertical Simulator")
        title_box.label.set_fontsize(24)

        text_box = TextBox(axbox, 'Mass(Kg)', initial=0.5)
        text_box1 = TextBox(axbox1, 'Spring constant(N/m)', initial=2.5)
        text_box2 = TextBox(axbox2, 'Amplitude(m)', initial=10)
        text_box3 = TextBox(axbox3, 'Damping Coefficient', initial=0.02)
        text_box4 = TextBox(axbox4, 'Gravitational acceleration', initial=9.8)
        startButton = Button(axbutton, "START")
        checkbutton = CheckButtons(check_box, ["Show Equilibrium Line"])


        checkbutton.on_clicked(self.set_show_equilibrium_line)
        startButton.on_clicked(self.button_pressed)
        title_box.on_clicked(self.do_nothing)
        text_box.on_submit(self.set_mass)
        text_box1.on_submit(self.set_spring_constant)
        text_box2.on_submit(self.set_amplitude)
        text_box3.on_submit(self.set_damping)
        text_box4.on_submit(self.set_gravity)




        plt.show()


    def set_show_equilibrium_line(self, text):
        self.show_equilibrium_line =True


    def set_gravity(self, text):
        self.gravity = float(text)

    def set_mass(self, text):
        self.mass = float(text)

    def set_spring_constant(self, text):
        self.spring_constant = float(text)

    def set_amplitude(self, text):
        self.amplitude = float(text)

    def set_damping(self, text):
        self.damping_coefficient = float(text)

    def button_pressed(self, event):
        plt.close()

    def do_nothing(self, event):
        pass










gui1 = GUI()
gui1.show_home()

if gui1.option == 1:
    gui1.show_gui()

    plt2 = Horizontal_spring_mass(mass=gui1.mass, spring_constant=gui1.spring_constant,
                                  amplitude=gui1.amplitude, damping_coefficient=gui1.damping_coefficient)

    plt2.animate()
    plt2.plot_all_points()

else:

    gui1.show_gui1()

    plt2 = Vertical_spring_mass(mass=gui1.mass, spring_constant=gui1.spring_constant,
                                  amplitude=gui1.amplitude, damping_coefficient=gui1.damping_coefficient,
                                  gravity=gui1.gravity, display_equilibrium=gui1.show_equilibrium_line)

    plt2.animate()
    plt2.plot_all_points()


time1 = np.arange(0,len(plt2.plot_y_position_total), 0.05)





#Code for envolope
#Calculate the magnitude of peaks to store in list
envolope1 = envolope()
peaks, peak_time = envolope1.find_peaks(plt2.plot_y_position_total, plt2.time_x_total, equvilibrium=-plt2.equilibrium_position)


#Calculate the damping ratio by applying linear regression to log of the amplitude peaks
lodpk = []
for i in peaks:
    lodpk.append(math.log(i + plt2.equilibrium_position))
lr = Linear_regression()
a, b = lr.bestFit(peak_time, lodpk)                 # B is the slope of regression that is ,b = damping ratio*-W




#plot the envolope
fig = plt.figure(1)
fig.set_size_inches(plt2.window_sizeX, plt2.window_sizeY)
fig.tight_layout()
plt.subplot(211)
plt.plot(peak_time, peaks)
plt.plot(plt2.time_x_total, plt2.plot_y_position_total)
plt.title("Displacement vs time (envolope)")



plt.subplot(212)
plt.title("Slope (logarithemic) = "+str(b))
plt.xlabel("Damping ratio = " + str(-b/plt2.angular_frequency))
plt.plot(peak_time, lodpk)

plt.show()


#Print the damping ratio
print("Damping ratio is equal to logarithemic slope divided by -w (angular frequency)")
print("Damping Ratio = " + str(-b/plt2.angular_frequency))
