import matplotlib.pyplot as plt
from graphviz import Digraph
from matplotlib.collections import PolyCollection

def draw_DAG(dag, name=None, view=False):
    graph_dir = "graphs/"
    if name is not None:
        dot = Digraph(name=graph_dir + name)
    else:
        dot = Digraph()

    for task in dag.subtasks:
        name = "{},A={},C={}".format(task.name, round(task.a, 3), round(task.c, 3))
        dot.node(name, name[0])

    for task in dag.subtasks:
        name = "{},A={},C={}".format(task.name, round(task.a, 3), round(task.c, 3))
        for child in task.children:
            cname = "{},A={},C={}".format(child.name, round(child.a, 3), round(child.c, 3))
            dot.edge(name, cname)

    dot.render(view=view)


def get_original_taskname(task):
    return task.name.split("|")[0]


def schedule_timeline(taskset, schedule):
    cats = dict([(task.name, i+1) for i, task in enumerate(taskset)])
    colormapping = dict([(name, "C{}".format(i-1)) for name, i in cats.items()])

    verts = []
    colors = []
    for d in schedule:
        s, e, t = d
        name = get_original_taskname(t)
        v = [
            (s, cats[name]-.4),
            (s, cats[name]+.4),
            (e, cats[name]+.4),
            (e, cats[name]-.4),
            (s, cats[name]-.4)
        ]
        verts.append(v)
        colors.append(colormapping[name])

    bars = PolyCollection(verts, facecolors=colors)

    fig, ax = plt.subplots()
    ax.add_collection(bars)
    ax.autoscale()

    ax.set_yticks(list(range(1, len(taskset) + 1)))
    ax.set_yticklabels([t.name for t in taskset])
    plt.show()


def resource_timeline(taskset, schedule):
    resources = list(sorted(list(set([r for task in taskset for r in task.resources]))))
    cats = dict([(r, i + 1) for i, r in enumerate(resources)])
    colormapping = dict([(t.name, "C{}".format(i)) for i, t in enumerate(taskset)])

    verts = []
    colors = []
    for d in schedule:
        task_start, _, t = d
        name = get_original_taskname(t)
        subtasks = [t] if not hasattr(t, 'subtasks') else t.subtasks
        offset = task_start
        for subtask in subtasks:
            s = offset + subtask.a
            e = s + subtask.c
            for r in subtask.resources:
                v = [
                    (s, cats[r] - .4),
                    (s, cats[r] + .4),
                    (e, cats[r] + .4),
                    (e, cats[r] - .4),
                    (s, cats[r] - .4)
                ]
                verts.append(v)
                colors.append(colormapping[name])

    bars = PolyCollection(verts, facecolors=colors)

    fig, ax = plt.subplots()
    ax.add_collection(bars)
    ax.autoscale()

    ax.set_yticks(list(range(1, len(resources) + 1)))
    ax.set_yticklabels([r for r in resources])
    plt.show()