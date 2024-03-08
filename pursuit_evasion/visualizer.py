import matplotlib.cm as cm
import matplotlib.pyplot as plt

HSV = cm.get_cmap("hsv")
inferno = cm.get_cmap("inferno")


def normalize_pos_dict(pos_dict, output_size=600):
    min_x = min(pos_dict.values(), key=lambda x: x[0])[0]
    max_x = max(pos_dict.values(), key=lambda x: x[0])[0]
    min_y = min(pos_dict.values(), key=lambda x: x[1])[1]
    max_y = max(pos_dict.values(), key=lambda x: x[1])[1]
    updated_pos = {node: (pos_dict[node][0] - min_x, pos_dict[node][1] - min_y) for node in pos_dict}
    updated_pos = {node: (updated_pos[node][0] / (max_x - min_x), updated_pos[node][1] / (max_y - min_y)) for node in pos_dict}
    updated_pos = {node: ((updated_pos[node][0] * 0.8 + 0.1) * output_size, (updated_pos[node][1] * 0.8 + 0.1) * output_size) for node in pos_dict}
    return updated_pos

def normalize_pos_dict_square(pos_dict, output_size=600):
    min_x = min(pos_dict.values(), key=lambda x: x[0])[0]
    max_x = max(pos_dict.values(), key=lambda x: x[0])[0]
    min_y = min(pos_dict.values(), key=lambda x: x[1])[1]
    max_y = max(pos_dict.values(), key=lambda x: x[1])[1]
    size = max(max_x - min_x, max_y - min_y)
    updated_pos = {node: (pos_dict[node][0] - min_x, pos_dict[node][1] - min_y) for node in pos_dict}
    updated_pos = {node: (updated_pos[node][0] / size, updated_pos[node][1] / size) for node in pos_dict}
    updated_pos = {node: ((updated_pos[node][0] * 0.8 + 0.1) * output_size, (updated_pos[node][1] * 0.8 + 0.1) * output_size) for node in pos_dict}
    return updated_pos

def neighbors(y, x):
    return (
        (y + 1, x),
        (y, x + 1),
        (y - 1, x),
        (y, x - 1)
    )

def draw_edge_with_buffer(image, start, end, color, type, edge_thickness, arrow_buffer):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = (dx ** 2 + dy ** 2) ** 0.5
    direction_x = dx / distance
    direction_y = dy / distance
    start_x = start[0] + arrow_buffer * direction_x
    start_y = start[1] + arrow_buffer * direction_y
    end_x = end[0] - arrow_buffer * direction_x
    end_y = end[1] - arrow_buffer * direction_y
    start_x = int(start_x)
    start_y = int(start_y)
    end_x = int(end_x)
    end_y = int(end_y)
    # if type == 'arrow':
    #     cv2.arrowedLine(image, (end_x, end_y), (start_x, start_y), color, edge_thickness, tipLength=0.5)
    # elif type == 'line':
    #     cv2.line(image, (start_x, start_y), (end_x, end_y), color, edge_thickness, lineType=cv2.LINE_AA)

def render_scene(
    agents,
    tasks,
    node_size=5,
    # image_size,
    # edge_colors=(120, 120, 120),
    # outline_colors=(50, 50, 50),
    # edge_thickness=3,
    # outline_thickness=3,
    # arrows=[]
):
    for tag, info in agents.items():
        position = info['xy']
        color = info['color']
        action_probs = info['action_probs']

        plt.plot(position[0], position[1], 'o', color=color, markersize=node_size)
        # Render arrows for action probabilities
        for i, prob in enumerate(action_probs):
            if prob > 0:
                dx, dy = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)][i]
                print(i, prob, dx, dy)
                plt.arrow(position[0], position[1], dx * prob, dy * prob, color=color, head_width=0.05, head_length=0.1)
    
    for tag, info in tasks.items():
        position = info['xy']
        color = info['color']

        plt.plot(position[0], position[1], 'o', color=color, markersize=node_size)
    
    # set equal aspect ratio
    plt.xlim(-0.5, 10.5)
    plt.ylim(-0.5, 10.5)
    plt.gca().set_aspect('equal')
    plt.show()

