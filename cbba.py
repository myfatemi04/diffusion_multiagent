# Implementation of Consensus-Based Bundle Algorithm (CBBA)
# Aims to solve the task allocation problem in a distributed manner.

import dataclasses


@dataclasses.dataclass
class Position:
    x: float
    y: float

@dataclasses.dataclass
class Task:
    id: str
    value: float
    position: Position

@dataclasses.dataclass
class Message:
    sender_id: str
    y: dict[str, float] # `sender_id`'s best knowledge of the bids for each task
    z: dict[str, str | None] # `sender_id`'s best knowledge of the assignment of tasks to agents
    s: dict[str, float] # timestamps when the last message was received; keyed by agent id
    neighbors: list['AgentSolutionState']

class AgentSolutionState:
    def __init__(self, position: Position, tasks: dict[str, Task], agents: list[str], agent_id: str):
        self.position = position
        # lists of tasks
        self.bundle: list[str] = []
        self.path: list[str] = []
        # task -> bid
        self.y: dict[str, float] = {task: 0 for task in tasks.keys()}
        # task -> assignee
        self.z: dict[str, str | None] = {task: None for task in tasks.keys()}
        # agent -> timestamp for which information was received from this agent.
        # i.e., "how stale is this agent's information w.r.t. my point of view?"
        self.s: dict[str, float] = {agent: 0 for agent in agents}
        self.tasks = tasks
        self.agents = agents
        self.agent_id = agent_id

    def calculate_path_value(self, path: list[str]) -> float:
        prev_pos = self.position
        value = 0

        for i in range(len(path)):
            task = self.tasks[path[i]]
            dist = ((prev_pos.x - task.position.x)**2 + (prev_pos.y - task.position.y)**2)**0.5
            value = value - dist + task.value
            prev_pos = task.position

        return value

    def ingest_messages(self, messages: list[Message], timestamp: float):
        """
        Process incoming messages.
        Returns a Boolean indicating whether bundle needs to be reconstructed.
        """
        me = self.agent_id
        received_messages_from = set()

        Znext = self.z.copy()
        Ynext = self.y.copy()
        Snext = self.s.copy()

        for message in messages:
            self.debug(f"Received message from {message.sender_id}")

            them = message.sender_id
            received_messages_from.add(them)

            # Update time vector
            for neighbor in message.neighbors:
                Snext[neighbor.agent_id] = max(Snext[neighbor.agent_id], message.s[neighbor.agent_id])
            Snext[them] = timestamp

            for task in self.tasks:
                self.debug("Processing task", task)
                their_believed_assignee = message.z[task]
                my_believed_assignee = self.z[task]

                # Row 1.
                # Sender believes they are assigned the task.
                if their_believed_assignee == them:
                    self.debug(f"Sender believes they are assigned task.")

                    # I believe I have the task.
                    if my_believed_assignee == me:
                        self.debug("... and I believe I am assigned task.")

                        # If the sender outbids me, they get the task.
                        if message.y[task] > self.y[task]:
                            self.debug("... but they outbid me.")

                            Znext[task] = them
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... and I outbid them.")

                            # If I outbid them, I keep the task.
                            pass
                    elif (my_believed_assignee == them) or (my_believed_assignee is None):
                        self.debug("... and I believe they are assigned task.")

                        # If they believe they have the task, and so do I, they get it.
                        # If they believe they have the task, and I do not think
                        # anyone has the task, they get it.
                        Znext[task] = them
                        Ynext[task] = message.y[task]
                    else:
                        self.debug("... and I believe", my_believed_assignee, "is assigned task.")

                        # I believe neither of us have the task, and instead *m* has the task.
                        # If the sender received a message from *m* more recently than I did,
                        # they take the task (because they have more up-to-date information).
                        # Or, if the sender outbids *m*, they take the task.
                        if message.s[my_believed_assignee] > self.s[my_believed_assignee] or message.y[task] > self.y[task]:
                            self.debug("... but they outbid", my_believed_assignee)

                            Znext[task] = them
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... but they did not outbid", my_believed_assignee)

                # Row 2.
                # Sender believes I have the task.
                elif their_believed_assignee == me:
                    self.debug(f"Sender believes I am assigned task.")

                    if my_believed_assignee == me:
                        # I also believe I have the task. No change.
                        self.debug("... and I believe I am assigned task.")
                        pass
                    elif my_believed_assignee == them:
                        # I believe they have the task. Reset.
                        self.debug("... and I believe they are assigned task.")
                        Znext[task] = None
                        Ynext[task] = 0
                    elif my_believed_assignee is None:
                        # I believe nobody has the task. No change.
                        self.debug("... and I believe nobody is assigned task.")
                        pass
                    else:
                        # I believe neither of has the task, and instead *m* has the task.
                        # If they received a message from *m* more recently than I did,
                        # I reset.
                        self.debug("... and I believe", my_believed_assignee, "is assigned task.")
                        if message.s[my_believed_assignee] > self.s[my_believed_assignee]:
                            self.debug("... but they received a message from", my_believed_assignee, "more recently than I did.")
                            Znext[task] = None
                            Ynext[task] = 0
                        else:
                            self.debug("... but I have the most recent information about", my_believed_assignee, "being assigned task.")

                # Row 4.
                # Sender believes nobody has the task.
                elif their_believed_assignee is None:
                    self.debug(f"Sender believes nobody is assigned task.")

                    if my_believed_assignee == me:
                        # I believe I have the task. No change.
                        self.debug("... and I believe I am assigned task. No change.")
                        pass
                    elif my_believed_assignee == them:
                        # I believe they have the task. Update.
                        self.debug("... and I believe they are assigned task. Updating.")
                        Znext[task] = message.z[task]
                        Ynext[task] = message.y[task]
                    elif my_believed_assignee is None:
                        # I believe nobody has the task. No change.
                        self.debug("... and I also believe nobody is assigned task. No change.")
                        pass
                    else:
                        # I believe neither of us have the task, and instead *m* has the task.
                        # If they received a message from *m* more recently than I did,
                        # I update (i.e. reset)
                        self.debug("... and I believe", my_believed_assignee, "is assigned task.")
                        if message.s[my_believed_assignee] > self.s[my_believed_assignee]:
                            self.debug("... but they received a message from", my_believed_assignee, "more recently than I did. Updating.")
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... but I have the most recent information about", my_believed_assignee, "being assigned task. No change.")

                # Row 3.
                # Sender believes neither of us has the task, and instead *m* has the task.
                else:
                    self.debug(f"Sender believes {their_believed_assignee} is assigned task.")
                    
                    if my_believed_assignee == me:
                        # I believe I have the task. If they received a message from *m* more recently than I did,
                        # and they outbid me, I update.
                        self.debug("... and I believe I am assigned task.")
                        if message.s[their_believed_assignee] > self.s[their_believed_assignee] and message.y[task] > self.y[task]:
                            self.debug("... but they received a message from", their_believed_assignee, "more recently than I did and/or outbid me. Updating.")
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... but I have the most recent information about", their_believed_assignee, "being assigned task and/or they did not outbid me. No change.")
                    elif my_believed_assignee == them:
                        # I believe they have the task. If they received a message from *m* more recently than I did,
                        # I update. Otherwise, I received a message from *m* more recently than they did, but they
                        # think they have the task. I will reset in this case.
                        self.debug("... and I believe they are assigned task.")
                        if message.s[their_believed_assignee] > self.s[their_believed_assignee]:
                            self.debug("... but they received a message from", their_believed_assignee, "more recently than I did. Updating.")
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... but I have the most recent information about", their_believed_assignee, "being assigned task. Resetting.")
                            Znext[task] = None
                            Ynext[task] = 0
                    elif my_believed_assignee == their_believed_assignee:
                        # I agree that neither of us have the task. If they received a message from *m* more recently than I did,
                        # I will update their bid, though.
                        self.debug("... and I agree.")
                        if message.s[their_believed_assignee] > self.s[their_believed_assignee]:
                            self.debug("... but they received a message from", their_believed_assignee, "more recently than I did. Updating.")
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... but I have the most recent information about", their_believed_assignee, "being assigned task. No change.")
                    elif my_believed_assignee is None:
                        # I believe nobody has the task. If they received a message from *m* more recently than I did,
                        # I will update to reflect that, though.
                        self.debug("... and I believe nobody is assigned task.")
                        if message.s[their_believed_assignee] > self.s[their_believed_assignee]:
                            self.debug("... but they received a message from", their_believed_assignee, "more recently than I did. Updating.")
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... but I have the most recent information about", their_believed_assignee, "being assigned task. No change.")
                    else:
                        # If I think neither I, them, or their believed person are assigned the task...
                        # If they received a message from the person they believed it is assigned to more recently than I did,
                        # I will check if they received a message from who *I* believe it is assigned to more recently than I did,
                        # or if the person they believe it is assigned to outbid the person I believe it is assigned to.
                        if message.s[their_believed_assignee] > self.s[their_believed_assignee]:
                            if message.s[my_believed_assignee] > self.s[my_believed_assignee] or message.y[task] > self.y[task]:
                                Znext[task] = message.z[task]
                                Ynext[task] = message.y[task]
                        # If they received a message from the person I believe it is after me, and I received a message
                        # from the person they believe it is before them, I will reset.
                        elif message.s[my_believed_assignee] > self.s[my_believed_assignee] and self.s[their_believed_assignee] > message.s[their_believed_assignee]:
                            Znext[task] = None
                            Ynext[task] = 0

        # Update the agent's state.
        # Check for conflicts.
        earliest_conflict_index = -1
        for i, bundle_task in enumerate(self.bundle):
            if self.z[bundle_task] != Znext[bundle_task] or self.y[bundle_task] != Ynext[bundle_task]:
                earliest_conflict_index = i
                break

        # If we experience conflicts, we need to update y and z.
        # Additionally, we need to rebuild the bundle.
        if earliest_conflict_index != -1:
            self.debug("Conflicts detected. Rebuilding bundle.")
            self.bundle, self.path, self.y, self.z = release_items_added_after_index(
                self.tasks,
                self.bundle,
                self.path,
                Ynext, Znext,
                earliest_conflict_index
            )
            self.s = Snext
            return True
        else:
            self.debug("No conflicts detected.")
            self.s = Snext
            self.y = Ynext
            self.z = Znext
            return False

    def calculate_best_path_insertion_point(self, path, task):
        current_path_value = self.calculate_path_value(path)
        best_marginal_value = 0
        best_insertion_point = -1
        for i in range(len(path) + 1):
            path_next = path[:i] + [task] + path[i:]
            marginal_value = self.calculate_path_value(path_next) - current_path_value
            if marginal_value > best_marginal_value:
                best_marginal_value = marginal_value
                best_insertion_point  = i

        return (best_insertion_point, best_marginal_value)

    def debug(self, *args):
        # print(f"[agent {self.agent_id}]", *args)
        pass

    def build_bundle(self, max_bundle_size: int):
        """
        Creates a bundle.
        """
        Znext = self.z.copy()
        Ynext = self.y.copy()
        bundle_next = self.bundle.copy()
        path_next = self.path.copy()
        while len(bundle_next) < max_bundle_size:
            best_task = None
            best_task_marginal_value = 0
            best_task_insertion_point = None
            for task in self.tasks:
                if task in bundle_next:
                    continue
                
                n, marginal_value = self.calculate_best_path_insertion_point(path_next, task)
                bid_value = self.y[task]

                # I am outbid.
                if bid_value >= marginal_value:
                    continue

                # Ynext[task] = marginal_value
                
                if marginal_value > best_task_marginal_value:
                    best_task = task
                    best_task_marginal_value = marginal_value
                    best_task_insertion_point = n
            
            if best_task is None:
                # I cannot outbid anyone for any task.
                break
            else:
                assert best_task_insertion_point is not None, "Inconsistency between best_task and best_task_insertion_point"

                self.debug("Best task is", best_task, "with marginal value", best_task_marginal_value, "and insertion point", best_task_insertion_point)

                bundle_next.append(best_task)
                path_next.insert(best_task_insertion_point, best_task)
                Ynext[best_task] = best_task_marginal_value
                Znext[best_task] = self.agent_id
        
        self.z = Znext
        self.y = Ynext
        self.bundle = bundle_next
        self.path = path_next

def release_items_added_after_index(tasks: dict[str, Task], bundle: list[str], path: list[str], y: dict[str, float], z: dict[str, str | None], index: int):
    # Remove items that occur at or after `index`
    Bnext = bundle[:index]
    # Filter path to only those that are remaining
    Pnext = [task for task in path if task in Bnext]
    # Reset y-values for tasks after the one that was removed
    Ynext = {
        task: 0.0 if task in bundle[index + 1:] else y[task]
        for task in tasks.keys()
    }
    # Reset z-values for tasks after the one that was removed
    Znext = {
        task: None if task in bundle[index + 1:] else z[task]
        for task in tasks.keys()
    }
    return (Bnext, Pnext, Ynext, Znext)

def display_agents(agents: list[AgentSolutionState]):
    # Print results
    for agent in agents:
        print(f"Agent {agent.agent_id}:")
        print(f"  Bundle: {agent.bundle}")
        print(f"  Path: {agent.path}")
        print(f"  Value: {agent.calculate_path_value(agent.path)}")
        print(f"  Y: {agent.y}")
        print(f"  Z: {agent.z}")
        print(f"  S: {agent.s}")
        print()

def render_agents(agents: list[AgentSolutionState], tasks: dict[str, Task]):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    for agent in agents:
        plt.scatter(agent.position.x, agent.position.y, c='b')
        plt.annotate(agent.agent_id, (agent.position.x, agent.position.y))
    for task in tasks.values():
        plt.scatter(task.position.x, task.position.y, c='r')
        plt.annotate(task.id, (task.position.x, task.position.y))
    for agent in agents:
        prev_pos = agent.position
        for task_id in agent.path:
            task = tasks[task_id]
            plt.plot([prev_pos.x, task.position.x], [prev_pos.y, task.position.y], c='b')
            prev_pos = task.position
    plt.show()

def solve_cbba():
    import random

    random.seed(1)

    def random_position():
        return Position(
            x=random.random(),
            y=random.random(),
        )

    n_agents = 50
    n_tasks = 50
    max_bundle_size = 1

    agent_ids = [
        f'agent_{i}' for i in range(1, n_agents + 1)
    ]
    task_ids = [
        f'task_{i}' for i in range(1, n_tasks + 1)
    ]
    tasks = {
        task_id: Task(
            id=task_id,
            value=5,
            position=random_position(),
        )
        for task_id in task_ids
    }
    agents = [
        AgentSolutionState(random_position(), tasks, agent_ids, agent_id)
        for agent_id in agent_ids
    ]
    agents_by_id = {
        agent.agent_id: agent
        for agent in agents
    }
    # Create initial bids
    for agent in agents:
        agent.build_bundle(max_bundle_size)
    
    # Iterative message-passing algorithm
    mp_type = 'global'
    # Global communication graph
    if mp_type == 'global':
        adjacency_matrix = {
            agent.agent_id: agents
            for agent in agents
        }
    # Cyclic communication graph
    elif mp_type == 'cyclic':
        adjacency_matrix = {}
        for i in range(len(agents)):
            adjacency_matrix[agent_ids[i]] = [
                agents[(i - 1) % len(agents)],
                agents[(i + 1) % len(agents)]
            ]
    else:
        raise ValueError(f"Unknown message-passing type {mp_type}")

    display_agents(agents)
    for timestep in range(40):
        n_rebuilds_required = 0
        print(f"Running iteration {timestep + 1} of message-passing algorithm.")

        has_revisions = False
        for agent in agents:
            # Collect messages
            inbox = []
            for neighbor in adjacency_matrix[agent.agent_id]:
                neighbor_id = neighbor.agent_id
                if neighbor == agent.agent_id:
                    continue
                inbox.append(Message(
                    sender_id=neighbor_id,
                    y=neighbor.y,
                    z=neighbor.z,
                    s=neighbor.s,
                    neighbors=adjacency_matrix[neighbor.agent_id]
                ))
            # Receive messages
            needs_revisions = agent.ingest_messages(inbox, timestep)
            if needs_revisions:
                n_rebuilds_required += 1
                # Rebuild bundle
                agent.build_bundle(max_bundle_size)
                has_revisions = True
        
        render_agents(agents, tasks)

        if not has_revisions:
            print("Converged!")
            break

        # Show the existing assignments
        print(n_rebuilds_required, "rebuilds required.")
    else:
        print("Did not converge.")

    assigned_tasks = set()
    for agent in agents:
        print(f"Agent {agent.agent_id} is assigned tasks {agent.path}")
        assigned_tasks.update(agent.path)
    
    unassigned_tasks = set(tasks.keys()) - assigned_tasks
    print(f"Unassigned tasks: {unassigned_tasks}")
    
if __name__ == '__main__':
    solve_cbba()
