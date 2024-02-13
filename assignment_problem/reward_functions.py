import numpy as np

def evaluate_assignment(choices, agent_locations, task_locations, movement_cost=1/100, aggregate=False):
    agent_task_distance = np.linalg.norm(
        agent_locations[:, None, :].repeat(len(task_locations), axis=1) -
        task_locations[None, :, :].repeat(len(agent_locations), axis=0),
        axis=2
    )
    # calculate a value for each agent
    # by default, agents get penalized for not finding a task
    agent_values = [-1.0] * len(agent_locations)
    for task_id in range(len(task_locations)):
        least_cost = None
        least_cost_agent = None
        for agent_id, choice in enumerate(choices):
            if choice == task_id:
                if least_cost is None or agent_task_distance[choice, task_id] < least_cost:
                    least_cost = agent_task_distance[choice, task_id]
                    least_cost_agent = agent_id
        if least_cost is not None:
            assert least_cost_agent is not None
            # 10 can be reconfigured to mean a decay rate
            # agent_values[least_cost_agent] = 1 * np.exp(-least_cost / 40)
            # give a reward of 1
            agent_values[least_cost_agent] = 1 # 1 * np.exp(-least_cost / 40)
    # calculate cost incurred by moving far
    if movement_cost != 0:
        for agent_id, choice in enumerate(choices):
            agent_values[agent_id] -= float(np.linalg.norm(agent_locations[agent_id] - task_locations[choice])) * movement_cost
    if aggregate:
        return sum(agent_values)
    else:
        return agent_values
