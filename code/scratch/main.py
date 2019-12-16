from task import Task
from protocol import Action, ActionDAG, Protocol
from resource import Resource
from schedule import BlockResourceEDFScheduler


def main():
    # Construct Protocols and resources
    resources = [Resource(name="R{}".format(i)) for i in range(4)]

    # Protocol 1
    a = Action(type="LinkGen", duration=1, resources=[resources[0]], parents=[])
    b = Action(type="LinkGen", duration=2, resources=[resources[1]], parents=[a])
    c = Action(type="LinkGen", duration=1, resources=[resources[2]], parents=[b])
    action_dag1 = ActionDAG(sources=[a], sinks=[c])
    p1 = Protocol(name="P1", action_dag=action_dag1)

    # Protocol 2
    a = Action(type="LinkGen", duration=1, resources=[resources[0]], parents=[])
    b = Action(type="LinkGen", duration=2, resources=[resources[3]], parents=[])
    c = Action(type="LinkGen", duration=1, resources=[resources[2]], parents=[a, b])
    action_dag2 = ActionDAG(sources=[a, b], sinks=[c])
    p2 = Protocol(name="P2", action_dag=action_dag2)

    # Protocol 3
    a = Action(type="LinkGen", duration=1, resources=[resources[0]], parents=[])
    b = Action(type="LinkGen", duration=2, resources=[resources[0]], parents=[a])
    c = Action(type="LinkGen", duration=1, resources=[resources[0]], parents=[b])
    action_dag3 = ActionDAG(sources=[a], sinks=[c])
    p3 = Protocol(name="P3", action_dag=action_dag3)

    protocols = [p1, p2, p3]

    deadlines = [5, 6, 7]

    tasks = []
    # Convert Protocols to Tasks
    for p, d in zip(protocols, deadlines):
        protocol_duration = p.get_duration()
        protocol_task = Task(C=protocol_duration, D=d, T=d, subtasks=...)
        tasks.append(protocol_task)

    scheduler = BlockResourceEDFScheduler()
    s = scheduler.schedule(tasks=tasks, resources=resources)
    import pdb
    pdb.set_trace()
