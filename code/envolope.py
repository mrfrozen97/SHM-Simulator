import matplotlib.pyplot as plt
import numpy as np
import math
import random
import pandas as pd





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


            # If the curve reaches zero reset the mzximum..............
            if time[i]>36.9 and time[i]<37.3:
                print(abs(temp_max))


            if ((self.displacement[i-1] >= self.equvilibrium and self.displacement[i]<=self.equvilibrium) or (self.displacement[i-1]<=equvilibrium and self.displacement[i]>=equvilibrium)
                ) and abs(temp_max)>=self.minimum_peak:
                if time[i] > 36.9 and time[i] < 37.3:
                    print("hello")

                self.peaks.append(temp_max + equvilibrium)
                self.peak_time.append(temp_time)
                temp_max = 0

        return self.peaks, self.peak_time








#Set some values
amplitude = 10
damping_ratio = 0.0017
mass = 0.25
k = 12.2526

angular_frequency = math.sqrt(k/mass)
damped_angular_frequency = angular_frequency * math.sqrt(1-damping_ratio**2)


#Create some data
dis = []
time = []
for t in np.arange(0, 200, 0.02):
    temp_y = (amplitude + random.random()*random.randint(-2, 2)) * math.exp(-1 * damping_ratio * angular_frequency * t) * math.cos(damped_angular_frequency * t)
    dis.append(temp_y)
    time.append(t)





data = pd.read_csv("spring_mass_air_run_2_right.csv")
plt.style.use('dark_background')
time1 = list(data['Time'])
accn1 = list(data['Acceleration z'])

time1 = time
accn1 = dis



#Apply the class methods
en = envolope()
peaks, peak_time = en.find_peaks(accn1, time1, equvilibrium=0, minimum_peak=0)


#Take log of values

ec = math.e
lodpk = []
for i in peaks:
    lodpk.append(math.log((i)/(amplitude*angular_frequency**2), ec))




#Apply linear regression to it
lr = Linear_regression()
a, b = lr.bestFit(peak_time, lodpk)                 # B is the slope of regression that is ,b = damping ratio*-W
print("Damping ratio = " + str(b/-angular_frequency))
print(lr.squared_error())
#print("a = " + str(a))


plt.subplot(211)
plt.plot(peak_time, peaks)
print(peaks)
#plt.plot([0, 500], [15500, 15500])
plt.plot(time1, accn1)



"""
#Plot the same.....................................
plt.style.use('dark_background')

plt.plot(peak_time, lodpk)
plt.plot(peak_time, peaks)
plt.plot(time, dis)
plt.show()
"""


plt.subplot(212)
damping_ratio = -b/angular_frequency
print(damping_ratio)
dis = []
time = []
for t in np.arange(0, 200, 0.1):
    temp_y = (amplitude + random.random() * random.randint(-2, 2)) * math.exp(-1 * damping_ratio * angular_frequency * t) * math.cos(damped_angular_frequency * t)
    dis.append(temp_y)
    time.append(t)


plt.plot(time, dis)
plt.show()

