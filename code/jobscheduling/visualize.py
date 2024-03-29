import matplotlib.pyplot as plt
from graphviz import Digraph
from matplotlib.collections import PolyCollection


def draw_DAG(dag, name=None, view=False):
    """
    Draws a DAG that represents the repeater protocol to be executed
    :param dag: type DAGTask
        The DAGTask representing the repeater protocol to execute
    :param name: type str
        Name to give to the graph file
    :param view: type bool
        Toggles if the drawn DAG is opened in a viewer
    :return: None
    """
    graph_dir = "graphs/"
    if name is not None:
        dot = Digraph(name=graph_dir + name)
    else:
        dot = Digraph()

    for task in dag.subtasks:
        name = "{},A={},C={}".format(task.name, round(task.a, 3), round(task.c, 3))
        action = name[0]
        dot.node(name, action)

    for task in dag.subtasks:
        name = "{},A={},C={}".format(task.name, round(task.a, 3), round(task.c, 3))
        for child in task.children:
            cname = "{},A={},C={}".format(child.name, round(child.a, 3), round(child.c, 3))
            dot.edge(name, cname)

    dot.render(view=view)


def get_original_taskname(task):
    """
    Obtains the original name of a task instance
    :param task: type DAGTask
        An instance of a Periodic DAG Task to obtain the original task name from
    :return: type str
        The original PeriodicDAGTask's name
    """
    return task.name.split("|")[0]


def schedule_timeline(taskset, schedule):
    """
    Produces a visualization of the timeline where repeater protocols are executed in the schedule
    :param taskset: type list
        List of PeriodicDAGTask's representing the repeater protocols scheduled into the network
    :param schedule: type list
        List of tuples (start, end, task_instance) that describes the schedule of repeater protocols in the network
    :return: None
    """
    cats = dict([(task.name, i + 1) for i, task in enumerate(taskset)])
    colormapping = dict([(name, "C{}".format(i - 1)) for name, i in cats.items()])

    verts = []
    colors = []
    for d in schedule:
        s, e, t = d
        name = get_original_taskname(t)
        v = [
            (s, cats[name] - .4),
            (s, cats[name] + .4),
            (e, cats[name] + .4),
            (e, cats[name] - .4),
            (s, cats[name] - .4)
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
    """
    Produces a visualization of the timeline where resources are used schedule
    :param taskset: type list
        List of PeriodicDAGTask's representing the repeater protocols scheduled into the network
    :param schedule: type list
        List of tuples (start, end, task_instance) that describes the schedule of repeater protocols in the network
    :return: None
    """
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
        for subtask in sorted(subtasks, key=lambda subtask: -subtask.c):
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


def protocol_timeline(scheduled_protocol_task):
    """
    Produces a visualization of the timeline of resource utilization by a protocol
    :param scheduled_protocol_task: type DAGTask
        The DAGTask to visualize on resources
    :return: None
    """
    resources = list(sorted(scheduled_protocol_task.resources))
    resource_intervals = scheduled_protocol_task.get_resource_intervals()
    cats = dict([(r, i + 1) for i, r in enumerate(resources)])
    colormapping = {
        "L": 'C0',
        "S": 'C1',
        "D": 'C2',
        "O": "C3"
    }

    verts = []
    colors = []
    for r in resources:
        for interval in resource_intervals[r]:
            s = interval.begin
            e = interval.end
            t = interval.data.name[0]
            v = [
                (s, cats[r] - .4),
                (s, cats[r] + .4),
                (e, cats[r] + .4),
                (e, cats[r] - .4),
                (s, cats[r] - .4)
            ]
            verts.append(v)
            colors.append(colormapping[t])

    bars = PolyCollection(verts, facecolors=colors)

    fig, ax = plt.subplots()
    ax.add_collection(bars)
    ax.autoscale()
    yticks = resources
    ax.set_yticks(list(range(1, len(resources) + 1)))
    ax.set_yticklabels(yticks)
    plt.show()


def schedule_and_resource_timelines(taskset, schedule, plot_title=None, plot_sep=True, save_plot=False):
    """
    Produces a visualization of the timeline where resources are used schedule and where repeater protocols are
    executed
    :param taskset: type list
        List of PeriodicDAGTask's representing the repeater protocols scheduled into the network
    :param schedule: type list
        List of tuples (start, end, task_instance) that describes the schedule of repeater protocols in the network
    :return: None
    """
    task_cats = dict([(task.name, i + 1) for i, task in enumerate(taskset)])
    task_colormapping = dict([(name, "C{}".format(i - 1)) for name, i in task_cats.items()])

    verts = []
    colors = []
    resources = set()
    for d in schedule:
        s, e, t = d
        resources |= set(t.resources)
        name = get_original_taskname(t)
        v = [
            (s, task_cats[name] - .4),
            (s, task_cats[name] + .4),
            (e, task_cats[name] + .4),
            (e, task_cats[name] - .4),
            (s, task_cats[name] - .4)
        ]
        verts.append(v)
        colors.append(task_colormapping[name])

    task_bars = PolyCollection(verts, facecolors=colors)

    resources = list(sorted(resources))
    resource_cats = dict([(r, i + 1) for i, r in enumerate(resources)])

    verts = []
    colors = []
    for d in schedule:
        task_start, task_end, t = d
        name = get_original_taskname(t)
        resource_intervals = t.get_resource_intervals()
        for r, itree in resource_intervals.items():
            itree.merge_overlaps(strict=False)
            for interval in itree:
                s = interval.begin + task_start - t.a
                e = min(task_end, interval.end + task_start - t.a)
                v = [
                    (s, resource_cats[r] - .4),
                    (s, resource_cats[r] + .4),
                    (e, resource_cats[r] + .4),
                    (e, resource_cats[r] - .4),
                    (s, resource_cats[r] - .4)
                ]
                verts.append(v)
                colors.append(task_colormapping[name])

    resource_bars = PolyCollection(verts, facecolors=colors)

    if plot_sep:
        fig, ax = plt.subplots()
        ax.add_collection(task_bars)
        ax.autoscale()

        ax.set_yticks(list(range(1, len(taskset) + 1)))
        ax.set_yticklabels([t.name for t in taskset])
        if plot_title:
            plt.title(plot_title)
        if save_plot and plot_title:
            plt.savefig(fname="{}_tasks.png".format(plot_title))
        else:
            plt.show()

        fig, ax = plt.subplots()
        ax.add_collection(resource_bars)
        ax.autoscale()

        # name_map = {'1,0': "Alice", '1,1': "Repeater", '1,2': "Bob", "0,1": "Charlie", "2,1": "David"}
        # yticks = [name_map[r[0:3]] + r[3:] for r in resources]
        yticks = resources
        ax.set_yticks(list(range(1, len(resources) + 1)))
        ax.set_yticklabels(yticks)
        if plot_title:
            plt.title(plot_title)
        if save_plot and plot_title:
            plt.savefig(fname="{}_resources.png".format(plot_title))
        else:
            plt.show()

    else:
        fig, axs = plt.subplots(2)
        axs[0].add_collection(task_bars)
        axs[0].autoscale()

        axs[0].set_yticks(list(range(1, len(taskset) + 1)))
        axs[0].set_yticklabels([t.name for t in taskset])

        axs[1].add_collection(resource_bars)
        axs[1].autoscale()

        axs[1].set_yticks(list(range(1, len(resources) + 1)))
        axs[1].set_yticklabels([r for r in resources])

        if plot_title:
            plt.title(plot_title)
        if save_plot and plot_title:
            plt.savefig(fname="{}.png".format(plot_title))
        else:
            plt.show()
