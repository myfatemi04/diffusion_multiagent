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
        dEdx = (data[2:, :] - data[:-2, :]) / 2
        dEdy = (data[:, 2:] - data[:, :-2]) / 2
        dEdx = np.pad(dEdx, ((1, 1), (0, 0)))
        dEdy = np.pad(dEdy, ((0, 0), (1, 1)))
        self.gradient_magnitude = np.pad(np.sqrt(dEdx**2 + dEdy**2), 1)

        for i in range(M):
            for j in range(N):
                if i + 1 < M:
                    G.edges[(i, j), (i+1, j)]['weight'] = max(noise[i][j], noise[i + 1][j])
                    G.edges[(i+1, j), (i, j)]['weight'] = max(noise[i][j], noise[i + 1][j])

                if j + 1 < N:
                    G.edges[(i, j), (i, j+1)]['weight'] = max(noise[i][j], noise[i][j + 1])
                    G.edges[(i, j+1), (i, j)]['weight'] = max(noise[i][j], noise[i][j + 1])

    def least_cost_path(self, start, goal):
        return nx.dijkstra_path(self.graph, start, goal)

noise = generate_perlin_noise_2d((100, 100), (5, 5))
noise -= noise.min()
hm = Heatmap(noise)

# Question: What is the shortest path from the top left to the bottom right?

plt.subplot(1, 2, 1)
plt.imshow(hm.gradient_magnitude)
plt.subplot(1, 2, 2)
plt.imshow(hm.data, cmap='gist_earth', vmin=-0.5)
plt.show()

a,b = 0,0
c,d = 99,99
# Get shortest path
fig = plt.figure()
plt.imshow(noise, cmap='gist_earth', vmin=-0.5)
path: list[tuple[int, int]] = hm.least_cost_path((a, b), (c, d)) # type: ignore
contour = plt.plot([x[1] for x in path], [x[0] for x in path], color='red', linewidth=2)

# From https://matplotlib.org/stable/users/explain/figure/event_handling.html
def onclick(event):
    global contour
    x = int(event.xdata)
    y = int(event.ydata)
    path: list[tuple[int, int]] = hm.least_cost_path((0, 0), (int(y), int(x))) # type: ignore
    cost = 0
    for i in range(len(path)-1):
        cost += hm.graph.edges[(path[i], path[i+1])]['weight']
    contour[0].remove()
    contour = plt.plot([x[1] for x in path], [x[0] for x in path], color='red', linewidth=2)
    fig.canvas.draw()
    print(cost)

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

