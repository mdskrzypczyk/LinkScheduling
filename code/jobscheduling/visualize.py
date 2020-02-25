from graphviz import Digraph


def draw_DAG(dag, name=None, view=False):
    graph_dir = "graphs/"
    if name is not None:
        dot = Digraph(name=graph_dir + name)
    else:
        dot = Digraph()

    for task in dag.subtasks:
        name = "{},A={},C={}".format(task.name, round(task.a, 3), round(task.c, 3))
        dot.node(name, name)

    for task in dag.subtasks:
        name = "{},A={},C={}".format(task.name, round(task.a, 3), round(task.c, 3))
        for child in task.children:
            cname = "{},A={},C={}".format(child.name, round(child.a, 3), round(child.c, 3))
            dot.edge(name, cname)

    dot.render(view=view)
