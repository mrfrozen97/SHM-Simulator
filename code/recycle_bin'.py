"""
class entity:

    def __init__(self):
        self.a = 4
        self.b = 1.33
        self.down = False
        self.window_sizeX = 7
        self.window_sizeY = 7

        self.x = np.arange(400)
        self.y = np.array([math.sin((self.a*(i*math.pi)/180) + self.b) for i in self.x])
        self.x_spring = np.array([i/40 for i in range(100, 635)])
        self.y_spring = np.array([math.sin((self.a * (i * math.pi) / 180)) for i in self.x_spring])


    def calculate(self, index):
        plt.clf()
        index = index*2
        self.y = np.array([math.sin((self.a * (i * math.pi) / 180) + self.b +index/5) for i in self.x])
        y_temp = np.array([math.cos((self.a * (i * math.pi) / 180) + self.b + index / 5) for i in self.x])


        plt.figure(1)

        plt.subplot(211)
        plt.plot(self.x, self.y, label='Velocity')
        plt.plot(self.x, y_temp, label='Accelaration')
        plt.legend()
        plt.yticks([-1, 0, 1, 2])
        plt.subplot(212)


        #distance = 5*math.cos(self.a*index/18 + self.b) + 7
        distance = 5 * math.cos((self.a * (math.pi) / 180) + self.b +index/5 + 1) + 7
        self.y_spring = np.array([math.sin(((i*160 * math.pi) / 180)*(distance/4)) + 5 for i in self.x_spring])/5

        temp_x = [x for x in self.x_spring if x>distance and x<16]
        temp_y = self.y_spring[:len(temp_x)]


        #print(len(temp_y) - len(temp_x))
        #print(temp_x)


        #plt.plot([1,1], [16, distance])

        # Code that draws rectangle....

        plt.plot([0, 2], [0+distance, 0+distance])
        plt.plot([0, 0], [0+distance, -2+distance])
        plt.plot([2, 2], [0+distance, -2+distance])
        plt.plot([0, 2], [-2+distance, -2+distance])

        plt.plot(temp_y, temp_x, zorder=1)
        plt.scatter([1], [distance], color='r', s=1000, zorder=2, alpha=1)


       # plt.plot(self.y_spring, self.x_spring)

        plt.yticks([2*i for i in range(-1, 8)])
        plt.xticks([2*i for i in range(-2, 4)])


        #plt.show()



    def animate(self):

        plt.style.use('dark_background')
        fig = plt.figure()


        plt.Circle((0, 0), radius=4, color='r')
        fig.set_size_inches(self.window_sizeX, self.window_sizeY)
        ani = FuncAnimation(fig, self.calculate, interval=50, frames=1000)
        plt.show()


plt1 = entity()
plt1.animate()


"""


n = int(input())

parent = [int(x) for x in input().split()]


q = int(input())
queries =[]
for i in range(q):
    queries.append([int(x) for x in input().split()])

child = [0 for i in range(n)]
child_list = [[] for i in range(n)]

for i in range(n-1):
    child[parent[i]-1]+=1
    child_list[parent[i]-1].append(i+1)

print(child)
print(child_list)



def query(node, count, child, answer, child_list):

    if child[node-1] == 0:
        return answer

    if count%child[node-1] == 0:
        temp = 0
        for i in range(child[node-1]):
            temp = temp + query(child_list[node-1][i]+1, count/child[node-1], child, answer, child_list)
        return answer + temp

    else:
        return answer + count

for i in queries:
    print(query(i[0], i[1], child, 0, child_list))

