import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def solve_one_task_per_agent(agent_start_positions, task_locations, task_values):
    """
    Note: This is equivalent to "linear sum assignment".
    Parameters:
     * `agent_start_positions`: [nA, 2]
     * `task_locations`: [nT, 2]
     * `task_values`: [nT]
    """

    """
    Parameters for `scipy.optimize.linprog`:
     - c: Cost = c^T @ x
     - A_ub, b_ub: A_ub @ x <= b_ub
     - A_eq, b_eq: A_eq @ x = b_eq
    """

    num_agents = len(agent_start_positions)
    num_tasks = len(task_locations)

    # Calculate travel costs.
    # Repeat `agent_start_positions` to create nT columns.
    # Repeat `task_locations` to create nA rows.
    # Subtract the two matrices to get distances.
    distances = np.linalg.norm(
        agent_start_positions[:, np.newaxis, :].repeat(num_tasks, axis=1) -
        task_locations[np.newaxis, :, :].repeat(num_agents, axis=0),
        axis=2
    )

    objective = np.zeros(num_agents * num_tasks)

    for i in range(num_agents):
        for j in range(num_tasks):
            ij = i * num_tasks + j
            objective[ij] = task_values[j] - distances[i, j]
    
    A_ub = np.zeros((num_agents + num_tasks, num_agents * num_tasks))
    b_ub = np.zeros(num_agents + num_tasks)

    for i in range(num_agents):
        ij_1 = i * num_tasks + 0
        ij_2 = i * num_tasks + num_tasks
        A_ub[i, ij_1:ij_2] = 1
        b_ub[i] = 1

    for j in range(num_tasks):
        ij_1 = 0 * num_tasks + j
        ij_2 = num_agents * num_tasks + j
        A_ub[num_agents + j, ij_1:ij_2:num_tasks] = 1
        b_ub[num_agents + j] = 1

    A_eq = None
    b_eq = None

    res = opt.linprog(-objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), integrality=1)

    return res, (objective, A_ub, b_ub, A_eq, b_eq)

def solve_multiple_tasks_per_agent(agent_start_positions, task_locations, task_values):
    num_tasks = len(task_locations)
    num_agents = len(agent_start_positions)
    distances_agent2task = np.linalg.norm(
        agent_start_positions[:, np.newaxis, :].repeat(num_tasks, axis=1) -
        task_locations[np.newaxis, :, :].repeat(num_agents, axis=0),
        axis=2
    )
    distances_task2task = np.linalg.norm(
        task_locations[:, np.newaxis, :].repeat(num_tasks, axis=1) -
        task_locations[np.newaxis, :, :].repeat(num_tasks, axis=0),
        axis=2
    )
    # Calculate objective function.
    objective = np.zeros((num_agents * (num_tasks + 1) * num_tasks))
    for agent in range(num_agents):
        for start_task in range(num_tasks + 1):
            for next_task in range(num_tasks):
                if start_task == 0:
                    objective[np.ravel_multi_index((agent, start_task, next_task), (num_agents, num_tasks + 1, num_tasks))] = task_values[next_task] - distances_agent2task[agent, next_task]
                else:
                    objective[np.ravel_multi_index((agent, start_task, next_task), (num_agents, num_tasks + 1, num_tasks))] = task_values[next_task] - distances_task2task[start_task - 1, next_task]
        pass
    
    # Create upper bounds.
    # Multiple entries impossible -> num_tasks constraints
    # Multiple exits impossible -> num_tasks constraints
    # Every non-first task (i.e., j > 0) must have at least 1 entry (that is, for task j, at least one G_{aij} or G_{a0j} must be true.)
    #  -> num_agents * num_tasks constraints

    A_ub = np.zeros((num_tasks + (num_tasks + 1), num_agents * (num_tasks + 1) * num_tasks))
    b_ub = np.zeros(num_tasks + (num_tasks + 1))

    # 1. Multiple entries impossible
    counter = 0
    for next_task in range(num_tasks):
        for agent in range(num_agents):
            for potential_entrypoint in range(num_tasks + 1):
                idx = np.ravel_multi_index((agent, potential_entrypoint, next_task), (num_agents, num_tasks + 1, num_tasks))
                A_ub[counter, idx] = 1
        b_ub[counter] = 1
        counter += 1
    
    # 2. Multiple exits impossible
    for start_task in range(num_tasks + 1):
        for agent in range(num_agents):
            for potential_exitpoint in range(num_agents):
                idx = np.ravel_multi_index((agent, start_task, potential_exitpoint), (num_agents, num_tasks + 1, num_tasks))
                A_ub[counter, idx] = 1
        b_ub[counter] = 1
        counter += 1

    A_eq = np.zeros((num_agents * num_tasks, num_agents * (num_tasks + 1) * num_tasks))
    b_eq = np.zeros(num_agents * num_tasks)
    
    # 3. Every non-first task must have at least 1 entry iff it is used.
    # That is, sum[G_{a(k+1)l}] = sum[G_{a0k}] + sum[G_{ajk}]
    counter = 0
    for agent in range(num_agents):
        for next_task in range(num_tasks):
            for start_task in range(num_tasks + 1):
                A_eq[counter, np.ravel_multi_index((agent, start_task, next_task), (num_agents, num_tasks + 1, num_tasks))] = 1

            for next_next_task in range(num_tasks):
                A_eq[counter, np.ravel_multi_index((agent, next_task + 1, next_next_task), (num_agents, num_tasks + 1, num_tasks))] = -1
            
            b_eq[counter] = 0
            counter += 1

    res = opt.linprog(-objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), integrality=1)

    return res, (objective, A_ub, b_ub, A_eq, b_eq)
    

def test_simple():
    agent_start_positions = np.array([
        [0, 0],
        [1, 1],
        [2, 2]
    ])
    task_locations = np.array([
        [1, 0],
        [0, 0.1],
        [2, 1]
    ])
    task_values = np.array([10, 10, 10])

    plt.title("Setup")
    plt.scatter(agent_start_positions[:, 0], agent_start_positions[:, 1], c='r', label='Agent Start Positions')
    plt.scatter(task_locations[:, 0], task_locations[:, 1], c='b', label='Task Locations')
    plt.legend()
    plt.show()

    res, matrices = solve_one_task_per_agent(agent_start_positions, task_locations, task_values)

    assignments = res.x.reshape((len(agent_start_positions), len(task_locations)))
    (assn_i, assn_j) = np.where(assignments > 0.5)

    plt.title("Solution")
    plt.scatter(agent_start_positions[:, 0], agent_start_positions[:, 1], c='r', label='Agent Start Positions')
    plt.scatter(task_locations[:, 0], task_locations[:, 1], c='b', label='Task Locations')
    plt.legend()
    for i, j in zip(assn_i, assn_j):
        print(f"Agent {i} assigned to Task {j} with value {task_values[j]}")
        plt.plot([agent_start_positions[i, 0], task_locations[j, 0]], [agent_start_positions[i, 1], task_locations[j, 1]], c='g')
    plt.show()

def test_randomized():
    num_agents = 10
    num_tasks = num_agents * 2

    agent_start_positions = np.random.rand(num_agents, 2)
    task_locations = np.random.rand(num_tasks, 2)
    task_values = np.ones(num_tasks) * 10 # np.random.rand(num_tasks) * 10

    plt.title("Setup")
    plt.scatter(agent_start_positions[:, 0], agent_start_positions[:, 1], c='r', label='Agent Start Positions')
    plt.scatter(task_locations[:, 0], task_locations[:, 1], c='b', label='Task Locations')
    plt.legend()
    plt.show()

    # res, matrices = solve_one_task_per_agent(agent_start_positions, task_locations, task_values)
    res, matrices = solve_multiple_tasks_per_agent(agent_start_positions, task_locations, task_values)

    assignments = res.x.reshape((len(agent_start_positions), len(task_locations) + 1, len(task_locations)))
    (assn_i, assn_j, assn_k) = np.where(assignments > 0.5)

    total_value = -res.fun
    print("Result:", res)
    print("Total value:", total_value)
    print(f"Num. Assignments Made: {len(assn_i)} / {num_tasks}")

    # plt.title("Solution")
    # plt.scatter(agent_start_positions[:, 0], agent_start_positions[:, 1], c='r', label='Agent Start Positions')
    # plt.scatter(task_locations[:, 0], task_locations[:, 1], c='b', label='Task Locations')
    # plt.legend()
    for i, j, k in zip(assn_i, assn_j,  assn_k):
        print(f"Agent {i}: {j} to {k + 1}")
    #     print(f"Agent {i} assigned to Task {j} with value {task_values[j]}")
    #     plt.plot([agent_start_positions[i, 0], task_locations[j, 0]], [agent_start_positions[i, 1], task_locations[j, 1]], c='g')

    # plt.show()

test_randomized()
