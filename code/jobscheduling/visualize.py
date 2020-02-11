from graphviz import Digraph


def draw_DAG(dag):
    dot = Digraph()

    for task in dag.subtasks:
        dot.node(task.name, task.name)

    for task in dag.subtasks:
        for child in task.children:
            dot.edge(task.name, child.name)

    dot.render('test_graph', view=True)
