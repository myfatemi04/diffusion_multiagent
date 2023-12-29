# Generate random terrain heatmap

import numpy as np
import matplotlib.pyplot as plt

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

noise = generate_perlin_noise_2d((100, 100), (5, 5))
noise -= noise.min()

# Question: What is the shortest path from the top left to the bottom right?
import networkx as nx

G = nx.grid_2d_graph(100, 100)
G = nx.DiGraph(G)

up_weight = 20
down_weight = 20
distance_weight = 0
complexity_weight = 0

dEdx = noise[2:, :] - noise[:-2, :]
dEdy = noise[:, 2:] - noise[:, :-2]
# dEdx2 = dEdx[1:, :] - dEdx[:-1, :]
# dEdy2 = dEdy[:, 1:] - dEdy[:, :-1]
complexity = np.pad(np.abs(dEdx), ((1, 1), (0, 0))) + np.pad(np.abs(dEdy), ((0, 0), (1, 1)))

plt.subplot(1, 2, 1)
plt.imshow(complexity)
plt.subplot(1, 2, 2)
plt.imshow(noise, cmap='gist_earth', vmin=-0.5)
plt.show()

noise2 = noise
noise = complexity
for i in range(100):
    for j in range(100):
        # if j < 99:
        #     G.edges[(i, j), (i, j+1)]['weight'] = max(0, noise[i][j + 1] - noise[i][j]) * up_weight - min(0, noise[i][j + 1] - noise[i][j]) * down_weight + distance_weight
        #     G.edges[(i, j+1), (i, j)]['weight'] = max(0, noise[i][j] - noise[i][j + 1]) * up_weight - min(0, noise[i][j] - noise[i][j + 1]) * down_weight + distance_weight

        # if i < 99:
        #     G.edges[(i, j), (i+1, j)]['weight'] = max(0, noise[i + 1][j] - noise[i][j]) * up_weight - min(0, noise[i + 1][j] - noise[i][j]) * down_weight + distance_weight
        #     G.edges[(i+1, j), (i, j)]['weight'] = max(0, noise[i][j] - noise[i + 1][j]) * up_weight - min(0, noise[i][j] - noise[i + 1][j]) * down_weight + distance_weight
        if j < 99:
            G.edges[(i, j), (i, j+1)]['weight'] = max(noise[i][j], noise[i][j + 1])
            G.edges[(i, j+1), (i, j)]['weight'] = max(noise[i][j], noise[i][j + 1])

        if i < 99:
            G.edges[(i, j), (i+1, j)]['weight'] = max(noise[i][j], noise[i + 1][j])
            G.edges[(i+1, j), (i, j)]['weight'] = max(noise[i][j], noise[i + 1][j])

        # if i < 99 and j < 99:
        #     G.edges[(i, j), (i+1, j+1)]['weight'] = noise[i + 1][j + 1] - noise[i][j]
        #     G.edges[(i+1, j+1), (i, j)]['weight'] = noise[i][j] - noise[i + 1][j + 1]

noise = noise2

a,b = 0,0
c,d = 99,99
# Get shortest path
fig = plt.figure()
plt.imshow(noise, cmap='gist_earth', vmin=-0.5)
path: list[tuple[int, int]] = nx.dijkstra_path(G, (a, b), (c, d)) # type: ignore
contour = plt.plot([x[1] for x in path], [x[0] for x in path], color='red', linewidth=2)

def onclick(event):
    global contour
    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #     ('double' if event.dblclick else 'single', event.button,
    #     event.x, event.y, event.xdata, event.ydata))
    x = int(event.xdata)
    y = int(event.ydata)
    path: list[tuple[int, int]] = nx.dijkstra_path(G, (0, 0), (int(y), int(x))) # type: ignore
    cost = 0
    for i in range(len(path)-1):
        cost += G.edges[(path[i], path[i+1])]['weight']
    contour[0].remove()
    contour = plt.plot([x[1] for x in path], [x[0] for x in path], color='red', linewidth=2)
    print(cost)
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

