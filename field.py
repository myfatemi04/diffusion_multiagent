# Generate random terrain heatmap

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# From https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html
def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

class Heatmap:
    def __init__(self, data: np.ndarray):
        assert len(data.shape) == 2, "Heatmap must be 2D"
        self.data = data
        self.shape = data.shape

        M, N = data.shape

        G = nx.grid_2d_graph(M, N)
        G = nx.DiGraph(G)
        self.graph = G

        # Use central approximation
        dataX = np.pad(data, ((1, 1), (0, 0)), mode='edge')
        dataY = np.pad(data, ((0, 0), (1, 1)), mode='edge')
        self.dEdx = (dataX[2:, :] - dataX[:-2, :]) / 2
        self.dEdy = (dataY[:, 2:] - dataY[:, :-2]) / 2
        self.gradient_magnitude = self.dEdx**2 + self.dEdy**2
        print(self.gradient_magnitude.min(), self.gradient_magnitude.max())

        for i in range(M):
            for j in range(N):
                if i + 1 < M:
                    G.edges[(i, j), (i+1, j)]['weight'] = max(self.gradient_magnitude[i][j], self.gradient_magnitude[i + 1][j]) + 0
                    G.edges[(i+1, j), (i, j)]['weight'] = max(self.gradient_magnitude[i][j], self.gradient_magnitude[i + 1][j]) + 0

                if j + 1 < N:
                    G.edges[(i, j), (i, j+1)]['weight'] = max(self.gradient_magnitude[i][j], self.gradient_magnitude[i][j + 1]) + 0
                    G.edges[(i, j+1), (i, j)]['weight'] = max(self.gradient_magnitude[i][j], self.gradient_magnitude[i][j + 1]) + 0

    def least_cost_path(self, start, goal):
        return nx.astar_path(self.graph, start, goal)

width = 400
height = 400
noise = generate_perlin_noise_2d((width, height), (8, 8))
noise -= noise.min()
noise = np.maximum(noise, 0.3)
hm = Heatmap(noise)

# Question: What is the shortest path from the top left to the bottom right?

ax1 = plt.subplot(1, 2, 1)
gm = hm.gradient_magnitude - hm.gradient_magnitude.min()
gm /= gm.max()
plt.imshow(gm, cmap='viridis')
plt.subplot(1, 2, 2)
plt.imshow(hm.data, cmap='gist_earth', vmin=-0.5)

a,b = 0,0
c,d = 99,99
# Get shortest path
fig = plt.gcf()
path: list[tuple[int, int]] = hm.least_cost_path((a, b), (c, d)) # type: ignore
contour = plt.plot([x[1] for x in path], [x[0] for x in path], color='red', linewidth=2)
contour2 = ax1.plot([x[1] for x in path], [x[0] for x in path], color='red', linewidth=2)

# From https://matplotlib.org/stable/users/explain/figure/event_handling.html
def onclick(event):
    global contour, contour2
    x = int(event.xdata)
    y = int(event.ydata)
    path: list[tuple[int, int]] = hm.least_cost_path((0, 0), (int(y), int(x))) # type: ignore
    cost = 0
    for i in range(len(path)-1):
        cost += hm.graph.edges[(path[i], path[i+1])]['weight']
    contour[0].remove()
    contour = plt.plot([x[1] for x in path], [x[0] for x in path], color='red', linewidth=2)
    contour2[0].remove()
    contour2 = ax1.plot([x[1] for x in path], [x[0] for x in path], color='red', linewidth=2)
    fig.canvas.draw()
    print(cost)

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.tight_layout()
plt.show()

