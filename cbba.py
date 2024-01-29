# Implementation of Consensus-Based Bundle Algorithm (CBBA)
# Aims to solve the task allocation problem in a distributed manner.

import dataclasses

import numpy as np


@dataclasses.dataclass
class Position:
    x: float
    y: float

@dataclasses.dataclass
class Task:
    id: str
    value: float
    decay_rate: float
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
        self.id = agent_id
        self.debug_flag = False
    
    def debug(self, *args):
        # if not self.debug_flag:
        #     return
        # if self.id in ['agent_9', 'agent_35']:
        # print(f"[agent {self.id}]", *args)
        pass

    def calculate_path_value(self, path: list[str]) -> float:
        prev_pos = self.position
        value = 0
        total_dist = 0

        for i in range(len(path)):
            task = self.tasks[path[i]]
            total_dist += ((prev_pos.x - task.position.x)**2 + (prev_pos.y - task.position.y)**2)**0.5
            value += task.value * np.exp(-total_dist * task.decay_rate)
            prev_pos = task.position

        return value

    def ingest_messages(self, messages: list[Message], timestamp: float):
        """
        Process incoming messages.
        Returns a Boolean indicating whether bundle needs to be reconstructed.
        """
        self.debug_flag = True
        self.debug("Current bundle:", self.bundle, [self.y[task] for task in self.bundle])
        me = self.id
        received_messages_from = set()

        Znext = self.z.copy()
        Ynext = self.y.copy()
        Snext = self.s.copy()

        for message in messages:
            self.debug_flag = False
            self.debug(f"Received message from {message.sender_id}")

            them = message.sender_id
            received_messages_from.add(them)

            # Update time vector
            for agent_id in self.agents:
                if agent_id != me:
                    Snext[agent_id] = max(Snext[agent_id], message.s[agent_id])
            Snext[them] = timestamp

            for task in self.tasks:
                if task == 'task_46':
                    self.debug_flag = True
                else:
                    self.debug_flag = False

                self.debug("Processing task", task)
                their_winner = message.z[task]
                their_winner_bid = message.y[task]
                my_winner = Znext[task]
                my_winner_bid = Ynext[task]

                # Row 1.
                # Sender believes they are assigned the task.
                if their_winner == them:
                    self.debug(f"Sender believes they are assigned task.")

                    # I believe I have the task.
                    if my_winner == me:
                        self.debug("... and I believe I am assigned task.")

                        # If the sender outbids me, they get the task.
                        if their_winner_bid > my_winner_bid:
                            self.debug(f"... but they outbid me [{their_winner_bid:.3f} vs. {my_winner_bid:.3f}]. Updating.")

                            Znext[task] = them
                            Ynext[task] = their_winner_bid
                        else:
                            self.debug("... and I outbid them. No change.")
                            pass
                    elif (my_winner == them) or (my_winner is None):
                        self.debug("... and I believe they are assigned task, or I do not have a prior belief. Updating.")

                        # If they believe they have the task, and so do I, they get it.
                        # If they believe they have the task, and I do not think
                        # anyone has the task, they get it.
                        Znext[task] = them
                        Ynext[task] = their_winner_bid
                    else:
                        self.debug("... and I believe", my_winner, "is assigned task.")

                        # I believe neither of us have the task, and instead *m* has the task.
                        # If the sender received a message from *m* more recently than I did,
                        # they take the task (because they have more up-to-date information).
                        # Or, if the sender outbids *m*, they take the task.
                        if message.s[my_winner] > Snext[my_winner] or their_winner_bid > my_winner_bid:
                            self.debug(f"... but they outbid {my_winner}. Updating.")

                            Znext[task] = them
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... but they did not outbid", my_winner)

                # Row 2.
                # Sender believes I have the task.
                elif their_winner == me:
                    self.debug(f"Sender believes I am assigned task.")

                    if my_winner == me:
                        # I also believe I have the task. No change.
                        self.debug("... and I believe I am assigned task. No change.")
                        pass
                    elif my_winner == them:
                        # I believe they have the task. Reset.
                        self.debug("... and I believe they are assigned task. Resetting.")
                        Znext[task] = None
                        Ynext[task] = 0
                    elif my_winner is None:
                        # I believe nobody has the task. No change.
                        self.debug("... and I believe nobody is assigned task. No change.")
                        pass
                    else:
                        # I believe neither of has the task, and instead *m* has the task.
                        # If they received a message from *m* more recently than I did,
                        # I reset.
                        self.debug("... and I believe", my_winner, "is assigned task.")
                        if message.s[my_winner] > Snext[my_winner]:
                            self.debug("... but they received a message from", my_winner, "more recently than I did. Resetting.")
                            Znext[task] = None
                            Ynext[task] = 0
                        else:
                            self.debug("... and I have the most recent information about", my_winner, "being assigned task.")

                # Row 4.
                # Sender believes nobody has the task.
                elif their_winner is None:
                    self.debug(f"Sender ({them}) believes nobody is assigned task.")

                    if my_winner == me:
                        # I believe I have the task. No change.
                        self.debug("... and I believe I am assigned task. No change.")
                        pass
                    elif my_winner == them:
                        # I believe they have the task. Update.
                        self.debug("... and I believe they are assigned task. Updating (Resetting).")
                        Znext[task] = message.z[task]
                        Ynext[task] = message.y[task]
                    elif my_winner is None:
                        # I believe nobody has the task. No change.
                        self.debug("... and I also believe nobody is assigned task. No change.")
                        pass
                    else:
                        # I believe neither of us have the task, and instead *m* has the task.
                        # If they received a message from *m* more recently than I did,
                        # I update (i.e. reset)
                        self.debug("... and I believe", my_winner, "is assigned task.")
                        if message.s[my_winner] > Snext[my_winner]:
                            self.debug("... but they received a message from", my_winner, "more recently than I did. Updating.")
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... but I have the most recent information about", my_winner, "being assigned task. No change.")

                # Row 3.
                # Sender believes neither of us has the task, and instead *m* has the task.
                else:
                    self.debug(f"Sender believes {their_winner} is assigned task.")
                    
                    if my_winner == me:
                        # I believe I have the task. If they received a message from *m* more recently than I did,
                        # and they outbid me, I update.
                        self.debug("... and I believe I am assigned task.")
                        if message.s[their_winner] > Snext[their_winner] and their_winner_bid > my_winner_bid:
                            self.debug("... but they received a message from", their_winner, "more recently than I did and/or outbid me. Updating.")
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... but I have the most recent information about", their_winner, "being assigned task and/or they did not outbid me. No change.")
                    elif my_winner == them:
                        # I believe they have the task. If they received a message from *m* more recently than I did,
                        # I update. Otherwise, I received a message from *m* more recently than they did, but they
                        # think they have the task. I will reset in this case.
                        self.debug("... and I believe they are assigned task.")
                        if message.s[their_winner] > Snext[their_winner]:
                            self.debug("... but they received a message from", their_winner, "more recently than I did. Updating.")
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... but I have the most recent information about", their_winner, "being assigned task. Resetting.")
                            Znext[task] = None
                            Ynext[task] = 0
                    elif my_winner == their_winner:
                        # I agree that neither of us have the task. If they received a message from *m* more recently than I did,
                        # I will update their bid, though.
                        self.debug("... and I agree.")
                        if message.s[their_winner] > Snext[their_winner]:
                            self.debug("... but they received a message from", their_winner, "more recently than I did. Updating.")
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... but I have the most recent information about", their_winner, "being assigned task. No change.")
                    elif my_winner is None:
                        # I believe nobody has the task. If they received a message from *m* more recently than I did,
                        # I will update to reflect that, though.
                        self.debug("... and I believe nobody is assigned task.")
                        if message.s[their_winner] > Snext[their_winner]:
                            self.debug("... but they received a message from", their_winner, "more recently than I did. Updating.")
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        else:
                            self.debug("... but I have the most recent information about", their_winner, "being assigned task. No change.")
                    else:
                        self.debug(f"... and I believe a third person ({my_winner}) is assigned task (with bid {my_winner_bid:.3f})")
                        # If I think neither I, them, or their believed person are assigned the task...
                        # If they received a message from the person they believed it is assigned to more recently than I did,
                        # I will check if they received a message from who *I* believe it is assigned to more recently than I did,
                        # or if the person they believe it is assigned to outbid the person I believe it is assigned to.
                        if message.s[their_winner] > Snext[their_winner] and message.s[my_winner] > Snext[my_winner]:
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        elif message.s[their_winner] > Snext[their_winner] and their_winner_bid > my_winner_bid:
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        # If they received a message from the person I believe it is after me, and I received a message
                        # from the person they believe it is before them, I will reset.
                        elif message.s[my_winner] > Snext[my_winner] and Snext[their_winner] > message.s[their_winner]:
                            self.debug("... but there is a conflict in who received which information first. Resetting.")
                            Znext[task] = None
                            Ynext[task] = 0
                        else:
                            self.debug("... but nothing was new. No change.")

            self.debug_flag = True

        # Update the agent's state.
        # Check for conflicts.
        earliest_conflict_index = -1
        for i, bundle_task in enumerate(self.bundle):
            if self.z[bundle_task] != Znext[bundle_task]:
                earliest_conflict_index = i
                break

        # If we experience conflicts, we need to update y and z.
        # Additionally, we need to rebuild the bundle.
        if earliest_conflict_index != -1:
            self.debug("Conflicts detected. Releasing tasks starting at index", earliest_conflict_index)
            self.bundle, self.path, self.y, self.z = release_items_added_after_index(
                self.id,
                self.tasks,
                self.bundle,
                self.path,
                Ynext,
                Znext,
                earliest_conflict_index,
            )
            self.debug("Bundle after release:", self.bundle, [self.y[task] for task in self.bundle])
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

    def build_bundle(self, max_bundle_size: int):
        """
        Creates a bundle.
        """
        Znext = self.z.copy()
        Ynext = self.y.copy()
        bundle_next = self.bundle.copy()
        path_next = self.path.copy()

        # Greedy algorithm.
        # Our greedy choice is to pick the item that gives the best marginal value,
        # among those that we are not outbid for and which are not already in our bundle.
        while len(bundle_next) < max_bundle_size:
            best_task = None
            best_task_marginal_value = 0
            best_task_insertion_point = None
            task_values = {}
            debug_agent = self.id == 'agent_1'
            for task in self.tasks:
                insertion_point, marginal_value = self.calculate_best_path_insertion_point(path_next, task)
                assert marginal_value > 0
                
                bid_value = Ynext[task]
                task_values[task] = marginal_value

                if task in bundle_next:
                    continue

                # I am outbid.
                if bid_value >= marginal_value:
                    # if debug_agent:
                    #     print("I am outbid for", task, "with bid value", bid_value, "and marginal value", marginal_value)
                    continue
                

                # Take argmax across all tasks to greedily choose
                # the one that gives the best marginal value.
                if marginal_value > best_task_marginal_value:
                    if debug_agent:
                        print("I am not outbid for", task, "with bid value", bid_value, "and marginal value", marginal_value, "and insertion point", insertion_point, "and path", path_next)
                    best_task = task
                    best_task_marginal_value = marginal_value
                    best_task_insertion_point = insertion_point

            # if debug_agent:
            #     print("Best task is", best_task, "with marginal value", best_task_marginal_value, "and insertion point", best_task_insertion_point)
            
            if best_task is None:
                # I cannot outbid anyone for any task.
                self.debug("No best task found.")
                break
            else:
                assert best_task_insertion_point is not None, "Inconsistency between best_task and best_task_insertion_point"

                # if debug_agent:
                #     self.debug("Best task is", best_task, "with marginal value", best_task_marginal_value, "and insertion point", best_task_insertion_point)

                bundle_next.append(best_task)
                path_next.insert(best_task_insertion_point, best_task)
                Ynext[best_task] = best_task_marginal_value
                Znext[best_task] = self.id

        # Assert DMG assumption
        bundle_values = [
            Ynext[task]
            for task in bundle_next
        ]
        for i in range(1, len(bundle_next)):
            earlier = bundle_next[i - 1]
            later = bundle_next[i]
            # if Ynext[earlier] < Ynext[later]:
            #     print(
            #         f"DMG assumption violated: {bundle_next} {bundle_values}"
            #     )
            #     print(earlier, later, self.id)
            #     print(Ynext[earlier], Ynext[later])
            #     print(task_values[earlier], task_values[later]) # type: ignore
            #     raise AssertionError("DMG Assumption Failed")
            
        self.debug("New bundle:", bundle_next, [Ynext[task] for task in bundle_next])
        
        self.z = Znext
        self.y = Ynext
        self.bundle = bundle_next
        self.path = path_next

def release_items_added_after_index(
    my_agent_id: str,
    tasks: dict[str, Task],
    bundle: list[str],
    path: list[str],
    y: dict[str, float],
    z: dict[str, str | None],
    index: int
):
    print("Releasing tasks; Current bundle:", bundle)
    print("Current Z[bundle]:", [z[task] for task in bundle])
    # Remove items that occur at or after `index`
    Bnext = bundle[:index]
    # Filter path to only those that are remaining
    Pnext = [task for task in path if task in Bnext]
    # Reset y-values for tasks after the one that was removed
    Ynext = {}
    Znext = {}
    for task in tasks:
        # Check if task is set to be released (i.e. after `index` in the bundle)
        # AND is still assigned to me. If it is not still assigned to me, we keep
        # the updated value. (This was based on a bit of debugging)
        if task in bundle[index + 1:] and z[task] == my_agent_id:
            Ynext[task] = 0.0
        else:
            Ynext[task] = y[task]

        # Reset all Z-values after the one that was removed
        if task in bundle[index + 1:] and z[task] == my_agent_id:
            Znext[task] = None
        else:
            Znext[task] = z[task]

    print("Corresponding Ynext, Znext:", Ynext, Znext)
    
    return (Bnext, Pnext, Ynext, Znext)

def display_agents(agents: list[AgentSolutionState]):
    # Print results
    for agent in agents:
        print(f"Agent {agent.id}:")
        print(f"  Bundle: {agent.bundle}")
        print(f"  Path: {agent.path}")
        print(f"  Value: {agent.calculate_path_value(agent.path)}")
        print(f"  Y: {agent.y}")
        print(f"  Z: {agent.z}")
        print(f"  S: {agent.s}")
        print()

def render_agents(agents: list[AgentSolutionState], tasks: dict[str, Task], special_tasks: list[str] = []):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    for agent in agents:
        plt.scatter(agent.position.x, agent.position.y, c='b')
        # plt.annotate(agent.id, (agent.position.x, agent.position.y))
    for task in tasks.values():
        plt.scatter(task.position.x, task.position.y, c='r' if task.id not in special_tasks else 'g')
        # plt.annotate(task.id, (task.position.x, task.position.y))
    for agent in agents:
        prev_pos = agent.position
        for i, task_id in enumerate(agent.path):
            task = tasks[task_id]
            cm = plt.get_cmap('viridis')
            plt.plot([prev_pos.x, task.position.x], [prev_pos.y, task.position.y], c=cm(i / len(agent.path)))
            prev_pos = task.position
    plt.show()

def solve_cbba():
    import random

    def random_position():
        return Position(
            x=random.random(),
            y=random.random(),
        )
    
    do_render = True

    # for seed in range(100000):
    for seed in [10067]:
        print("SEED:", seed)
        random.seed(seed)
        n_agents = 50
        max_bundle_size = 2
        n_tasks = n_agents * (max_bundle_size + 2)
        use_high_value_tasks = True

        agent_ids = [
            f'agent_{i}' for i in range(1, n_agents + 1)
        ]
        task_ids = [
            f'task_{i}' for i in range(1, n_tasks + 1)
        ]
        special_tasks = []
        tasks = {}
        for task_id in task_ids:
            if random.random() < 0.4:
                # High-value, front line-ish tasks
                if use_high_value_tasks:
                    special_tasks.append(task_id)
                    tasks[task_id] = Task(
                        id=task_id,
                        value=2.0,
                        decay_rate=0.2,
                        position=Position(
                            x=random.random() * 0.2 + 0.8,
                            y=random.random(),
                        ),
                    )
                else:
                    tasks[task_id] = Task(
                        id=task_id,
                        value=1.0,
                        decay_rate=0.1,
                        position=Position(
                            x=random.random() * 0.2 + 0.8,
                            y=random.random(),
                        ),
                    )
            else:
                # Lower-value, auxiliary tasks
                tasks[task_id] = Task(
                    id=task_id,
                    value=1.0,
                    decay_rate=0.1,
                    position=Position(
                        x=random.random() * 0.6 + 0.2,
                        y=random.random(),
                    ),
                )
        agents = [
            AgentSolutionState(Position(
                x=random.random() * 0.2,
                y=random.random(),
            ), tasks, agent_ids, agent_id)
            for agent_id in agent_ids
        ]
        agents_by_id = {
            agent.id: agent
            for agent in agents
        }
        if do_render:
            render_agents(agents, tasks, special_tasks)

        # Create initial bids
        for agent in agents:
            agent.build_bundle(max_bundle_size)

        # Calculate marginal values
        if not do_render:
            a = agents_by_id['agent_1']
            tA = 'task_2'
            tB = 'task_6'
            print(a.calculate_best_path_insertion_point([], tA))
            print(a.calculate_best_path_insertion_point([tB], tA))
            print(a.calculate_best_path_insertion_point([], tB))
            print(a.calculate_best_path_insertion_point([tA], tB))
            input("Press enter to continue...")

        # Iterative message-passing algorithm
        mp_type = 'global'
        # Global communication graph
        if mp_type == 'global':
            adjacency_matrix = {
                agent.id: agents
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

        # display_agents(agents)
        convergence_streak = 0
        message_counter = 0
        for timestep in range(1000 + n_agents):
            rebuilds: set[AgentSolutionState] = set()

            # pairs = []
            # for agent in agents:
            #     for neighbor in adjacency_matrix[agent.id]:
            #         if agent.id < neighbor.id:
            #             pairs.append((agent, neighbor))
            
            # # Randomize order of message passing
            # random.shuffle(pairs)

            # # Exchange messages
            # for agent, neighbor in pairs:
            #     to_neighbor = Message(
            #         sender_id=neighbor.id,
            #         y=neighbor.y,
            #         z=neighbor.z,
            #         s=neighbor.s,
            #         neighbors=adjacency_matrix[neighbor.id]
            #     )
            #     from_neighbor = Message(
            #         sender_id=agent.id,
            #         y=agent.y,
            #         z=agent.z,
            #         s=agent.s,
            #         neighbors=adjacency_matrix[agent.id]
            #     )
            #     if agent.ingest_messages([from_neighbor], message_counter):
            #         rebuilds.add(agent)
            #         agent.build_bundle(max_bundle_size)
            #     if neighbor.ingest_messages([to_neighbor], message_counter):
            #         rebuilds.add(neighbor)
            #         neighbor.build_bundle(max_bundle_size)
            #     message_counter += 1

            agents_order = agents.copy()
            random.shuffle(agents_order)

            message_counter += 1
            for agent in agents_order:
                # Broadcast to neighbors
                for neighbor in adjacency_matrix[agent.id]:
                    if neighbor.id == agent.id:
                        continue
                    # message = Message(
                    #     sender_id=neighbor_id,
                    #     y=neighbor.y,
                    #     z=neighbor.z,
                    #     s=neighbor.s,
                    #     neighbors=adjacency_matrix[neighbor.id]
                    # )
                    message = Message(
                        sender_id=agent.id,
                        y=agent.y,
                        z=agent.z,
                        s=agent.s,
                        neighbors=adjacency_matrix[agent.id]
                    )
                    if neighbor.ingest_messages([message], message_counter):
                        rebuilds.add(neighbor)
                        neighbor.build_bundle(max_bundle_size)
                # # Receive messages
                # needs_revisions = agent.ingest_messages(inbox, message_counter)
                # if needs_revisions:
                #     rebuilds.add(agent)
                #     # Rebuild bundle
                #     agent.build_bundle(max_bundle_size)
            
            if len(rebuilds) == 0:
                # Run algorithm at least n_agents more times to ensure that communication is complete.
                convergence_streak += 1
                print(f"[Step {timestep}] No disagreements ({convergence_streak} / {n_agents})")
                if convergence_streak >= n_agents:
                    break
            else:
                # Show the existing assignments
                print(f"[Step {timestep}]: {len(rebuilds)} disagreements.")
                convergence_streak = 0
        else:
            if convergence_streak > 0:
                print(f"[Result]: Incomplete convergence: {convergence_streak} / {n_agents}")
            else:
                print("[Result]: Did not converge to conflict-free solution.")

        assigned_tasks = set()
        for agent in agents:
            print(f"Agent {agent.id} is assigned tasks {agent.path}")
            assigned_tasks.update(agent.path)
        
        unassigned_tasks = set(tasks.keys()) - assigned_tasks
        print(f"Unassigned tasks: {unassigned_tasks}")

        if do_render:
            render_agents(agents, tasks, special_tasks)

            # Using Linear Sum Assignment
            import numpy as np
            import scipy.optimize
            mat = np.zeros((n_agents, n_tasks))
            for i, agent in enumerate(agents):
                for j, task_id in enumerate(task_ids):
                    mat[i, j] = -agent.calculate_path_value([task_id])
            
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(mat)
            print(row_ind, col_ind)
            for i, agent in enumerate(agents):
                agent.path = [task_ids[col_ind[i]]]
                agent.bundle = [task_ids[col_ind[i]]]
            render_agents(agents, tasks, special_tasks)
    
if __name__ == '__main__':
    solve_cbba()
