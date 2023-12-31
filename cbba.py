# Implementation of Consensus-Based Bundle Algorithm (CBBA)
# Aims to solve the task allocation problem in a distributed manner.

import dataclasses
import numpy as np

@dataclasses.dataclass
class Message:
    sender_id: str
    y: dict # `sender_id`'s best knowledge of the bids for each task
    z: dict # `sender_id`'s best knowledge of the assignment of tasks to agents
    s: dict # timestamps when the last message was received; keyed by agent id
    neighbors: list

class AgentSolutionState:
    def __init__(self, tasks: list, agents: list, agent_id: str):
        self.bundle = []
        self.path = []
        self.y: dict[str, float | None] = {task: None for task in tasks}
        self.z: dict[str, str | None] = {task: None for task in tasks}
        self.s: dict[str, float] = {agent: 0 for agent in agents}
        self.tasks = tasks
        self.agents = agents
        self.agent_id = agent_id

    def calculate_path_value(self, path) -> float:
        raise NotImplementedError()

    def ingest_messages(self, messages: list[Message], timestamp: float):
        """
        Process incoming messages and decide whether bundle needs to
        be reconstructed.
        """
        receiver_id = self.agent_id
        received_messages_from = set()

        Znext = self.z.copy()
        Ynext = self.y.copy()
        Snext = self.s.copy()

        for message in messages:
            sender_id = message.sender_id
            received_messages_from.add(sender_id)

            # Update time vector
            for neighbor in message.neighbors:
                Snext[neighbor] = max(Snext[neighbor], message.s[neighbor])
            Snext[sender_id] = timestamp

            for task in self.tasks:
                # Row 1.
                # Sender believes they are assigned the task.
                if message.z[task] == sender_id:
                    believed_assignee = self.z[task]
                    # I believe I have the task.
                    if believed_assignee == receiver_id:
                        # If the sender outbids me, they get the task.
                        if message.y[task] > self.y[task]:
                            Znext[task] = sender_id
                            Ynext[task] = message.y[task]
                    elif (believed_assignee == sender_id) or (believed_assignee is None):
                        # If they believe they have the task, and so do I, they get it.
                        # If they believe they have the task, and I do not think
                        # anyone has the task, they get it.
                        Znext[task] = sender_id
                        Ynext[task] = message.y[task]
                    else:
                        # I believe neither of us have the task, and instead *m* has the task.
                        # If the sender received a message from *m* more recently than I did,
                        # they take the task (because they have more up-to-date information).
                        # Or, if the sender outbids *m*, they take the task.
                        if message.s[believed_assignee] > self.s[believed_assignee] or message.y[task] > self.y[task]:
                            Znext[task] = sender_id
                            Ynext[task] = message.y[task]

                # Row 2.
                # Sender believes I have the task.
                elif message.z[task] == receiver_id:
                    if self.z[task] == receiver_id:
                        # I also believe I have the task. No change.
                        pass
                    elif self.z[task] == sender_id:
                        # I believe they have the task. Reset.
                        Znext[task] = None
                        Ynext[task] = None
                    elif self.z[task] is None:
                        # I believe nobody has the task. No change.
                        pass
                    else:
                        # I believe neither of has the task, and instead *m* has the task.
                        believed_assignee = self.z[task]
                        # If they received a message from *m* more recently than I did,
                        # I reset.
                        if message.s[believed_assignee] > self.s[believed_assignee]:
                            Znext[task] = None
                            Ynext[task] = None

                # Row 4.
                # Sender believes nobody has the task.
                elif message.z[task] is None:
                    if self.z[task] == receiver_id:
                        # I believe I have the task. No change.
                        pass
                    elif self.z[task] == sender_id:
                        # I believe they have the task. Update.
                        Znext[task] = message.z[task]
                        Ynext[task] = message.y[task]
                    elif self.z[task] is None:
                        # I believe nobody has the task. No change.
                        pass
                    else:
                        # I believe neither of us have the task, and instead *m* has the task.
                        believed_assignee = self.z[task]
                        # If they received a message from *m* more recently than I did,
                        # I update (i.e. reset)
                        if message.s[believed_assignee] > self.s[believed_assignee]:
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]

                # Row 3.
                # Sender believes neither of us has the task, and instead *m* has the task.
                else:
                    believed_assignee = message.z[task]
                    N = self.z[task]
                    if N == receiver_id:
                        # I believe I have the task. If they received a message from *m* more recently than I did,
                        # and they outbid me, I update.
                        if message.s[believed_assignee] > self.s[believed_assignee] and message.y[task] > self.y[task]:
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                    elif N == sender_id:
                        # I believe they have the task. If they received a message from *m* more recently than I did,
                        # I update. Otherwise, I received a message from *m* more recently than they did, but they
                        # think they have the task. I will reset in this case.
                        if message.s[believed_assignee] > self.s[believed_assignee]:
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                        else:
                            Znext[task] = None
                            Ynext[task] = None
                    elif N == believed_assignee:
                        # I agree that neither of us have the task. If they received a message from *m* more recently than I did,
                        # I will update their bid, though.
                        if message.s[believed_assignee] > self.s[believed_assignee]:
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                    elif N is None:
                        # I believe nobody has the task. If they received a message from *m* more recently than I did,
                        # I will update to reflect that, though.
                        if message.s[believed_assignee] > self.s[believed_assignee]:
                            Znext[task] = message.z[task]
                            Ynext[task] = message.y[task]
                    else:
                        # If I think neither I, them, or their believed person are assigned the task...
                        # If they received a message from the person they believed it is assigned to more recently than I did,
                        # I will check if they received a message from who *I* believe it is assigned to more recently than I did,
                        # or if the person they believe it is assigned to outbid the person I believe it is assigned to.
                        if message.s[believed_assignee] > self.s[believed_assignee]:
                            if message.s[N] > self.s[N] or message.y[task] > self.y[task]:
                                Znext[task] = message.z[task]
                                Ynext[task] = message.y[task]
                        # If they received a message from the person I believe it is after me, and I received a message
                        # from the person they believe it is before them, I will reset.
                        elif message.s[N] > self.s[N] and self.s[believed_assignee] > message.s[believed_assignee]:
                            Znext[task] = None
                            Ynext[task] = None

        # Update the agent's state
        self.z = Znext
        self.y = Ynext
        self.s = Snext

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

def solve_cbba():
    pass
