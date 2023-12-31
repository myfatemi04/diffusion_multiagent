# Implementation of Consensus-Based Bundle Algorithm (CBBA)
# Aims to solve the task allocation problem in a distributed manner.

import dataclasses
import numpy as np

@dataclasses.dataclass
class Message:
    sender_id: str
    y: dict[str, float] # `sender_id`'s best knowledge of the bids for each task
    z: dict[str, str | None] # `sender_id`'s best knowledge of the assignment of tasks to agents
    s: dict[str, float] # timestamps when the last message was received; keyed by agent id
    neighbors: list['AgentSolutionState']

class AgentSolutionState:
    def __init__(self, tasks: list[str], agents: list[str], agent_id: str):
        self.bundle = []
        self.path = []
        self.y: dict[str, float] = {task: 0 for task in tasks}
        self.z: dict[str, str | None] = {task: None for task in tasks}
        self.s: dict[str, float] = {agent: 0 for agent in agents}
        self.tasks = tasks
        self.agents = agents
        self.agent_id = agent_id

    def calculate_path_value(self, path) -> float:
        raise NotImplementedError()

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
            them = message.sender_id
            received_messages_from.add(them)

            # Update time vector
            for neighbor in message.neighbors:
                Snext[neighbor.agent_id] = max(Snext[neighbor.agent_id], message.s[neighbor.agent_id])
            Snext[them] = timestamp

            for task in self.tasks:
                their_believed_assignee = message.z[task]
                my_believed_assignee = self.z[task]

                # Row 1.
                # Sender believes they are assigned the task.
                if their_believed_assignee == them:
                    # I believe I have the task.
                    if my_believed_assignee == me:
                        # If the sender outbids me, they get the task.
                        if message.y[task] > self.y[task]:
                            Znext[task] = them
                            Ynext[task] = message.y[task]
                    elif (my_believed_assignee == them) or (my_believed_assignee is None):
                        # If they believe they have the task, and so do I, they get it.
                        # If they believe they have the task, and I do not think
                        # anyone has the task, they get it.
                        Znext[task] = them
                        Ynext[task] = message.y[task]
                    else:
                        # I believe neither of us have the task, and instead *m* has the task.
                        # If the sender received a message from *m* more recently than I did,
                        # they take the task (because they have more up-to-date information).
                        # Or, if the sender outbids *m*, they take the task.
                        if message.s[my_believed_assignee] > self.s[my_believed_assignee] or message.y[task] > self.y[task]:
                            Znext[task] = them
                            Ynext[task] = message.y[task]

                # Row 2.
                # Sender believes I have the task.
                elif their_believed_assignee == me:
                    if my_believed_assignee == me:
                        # I also believe I have the task. No change.
                        pass
                    elif my_believed_assignee == them:
                        # I believe they have the task. Reset.
                        Znext[task] = None
                        Ynext[task] = 0
                    elif my_believed_assignee is None:
                        # I believe nobody has the task. No change.
                        pass
                    else:
                        # I believe neither of has the task, and instead *m* has the task.
                        # If they received a message from *m* more recently than I did,
                        # I reset.
                        if message.s[my_believed_assignee] > self.s[my_believed_assignee]:
                            Znext[task] = None
                            Ynext[task] = 0

                # Row 4.
                # Sender believes nobody has the task.
                elif their_believed_assignee is None:
                    if my_believed_assignee == me:
                        # I believe I have the task. No change.
                        pass
                    elif my_believed_assignee == them:
                        # I believe they have the task. Update.
                        Znext[task] = message.z[task]
                        Ynext[task] = message.y[task]
                    elif my_believed_assignee is None:
                        # I believe nobody has the task. No change.
                        pass
                    else:
                        # I believe neither of us have the task, and instead *m* has the task.
                        # If they received a message from *m* more recently than I did,
                        # I update (i.e. reset)
                        if message.s[my_believed_assignee] > self.s[my_believed_assignee]:
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]

                # Row 3.
                # Sender believes neither of us has the task, and instead *m* has the task.
                else:
                    if my_believed_assignee == me:
                        # I believe I have the task. If they received a message from *m* more recently than I did,
                        # and they outbid me, I update.
                        if message.s[their_believed_assignee] > self.s[their_believed_assignee] and message.y[task] > self.y[task]:
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                    elif my_believed_assignee == them:
                        # I believe they have the task. If they received a message from *m* more recently than I did,
                        # I update. Otherwise, I received a message from *m* more recently than they did, but they
                        # think they have the task. I will reset in this case.
                        if message.s[their_believed_assignee] > self.s[their_believed_assignee]:
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        else:
                            Znext[task] = None
                            Ynext[task] = 0
                    elif my_believed_assignee == my_believed_assignee:
                        # I agree that neither of us have the task. If they received a message from *m* more recently than I did,
                        # I will update their bid, though.
                        if message.s[their_believed_assignee] > self.s[their_believed_assignee]:
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                    elif my_believed_assignee is None:
                        # I believe nobody has the task. If they received a message from *m* more recently than I did,
                        # I will update to reflect that, though.
                        if message.s[their_believed_assignee] > self.s[their_believed_assignee]:
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
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
                if bid_value is not None and bid_value >= marginal_value:
                    continue
                
                if marginal_value > best_task_marginal_value:
                    best_task = task
                    best_task_marginal_value = marginal_value
                    best_task_insertion_point = n
            
            if best_task is None:
                # I cannot outbid anyone for any task.
                break
            else:
                assert best_task_insertion_point is not None, "Inconsistency between best_task and best_task_insertion_point"

                bundle_next.append(best_task)
                path_next.insert(best_task_insertion_point, best_task)
                Ynext[best_task] = best_task_marginal_value
                Znext[best_task] = self.agent_id
        
        self.z = Znext
        self.y = Ynext
        self.bundle = bundle_next
        self.path = path_next

def release_items_added_after_index(tasks: list[str], bundle: list[str], path: list[str], y: dict[str, float], z: dict[str, str | None], index: int):
    # Remove items that occur after `index`
    Bnext = bundle[:index]
    # Preserve order of self.path
    Pnext = [task for task in path if task in Bnext]
    # Reset y-values for tasks that are no longer in the bundle
    Ynext = {
        task: y[task] if task in Bnext else 0.0
        for task in tasks
    }
    # Reset z-values for tasks that are no longer in the bundle
    Znext = {
        task: z[task] if task in Bnext else None
        for task in tasks
    }
    return (Bnext, Pnext, Ynext, Znext)

def solve_cbba():
    agent_ids = [
        f'agent_{i}' for i in range(1, 11)
    ]
    task_ids = [
        f'task_{i}' for i in range(1, 11)
    ]
    agents = [
        AgentSolutionState(task_ids, agent_ids, agent_id)
        for agent_id in agent_ids
    ]
    agents_by_id = {
        agent.agent_id: agent
        for agent in agents
    }
    # Create initial bids
    max_bundle_size = 1
    for agent in agents:
        agent.build_bundle(max_bundle_size)
    
    # Iterative message-passing algorithm
    # Global communication graph
    adjacency_matrix = {
        agent.agent_id: agents
        for agent in agents
    }
    for timestep in range(100):
        has_revisions = False
        for agent in agents:
            # Collect messages
            inbox = []
            for neighbor_id in agent.agents:
                if neighbor_id == agent.agent_id:
                    continue
                neighbor = agents_by_id[neighbor_id]
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
                # Rebuild bundle
                agent.build_bundle(max_bundle_size)
                has_revisions = True
        
        if not has_revisions:
            print("Converged!")
            break
    else:
        print("Did not converge.")
