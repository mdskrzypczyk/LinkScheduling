class Action:
    def __init__(self, type, duration, resources, parents):
        self.type = type
        self.duration = duration
        self.resources = resources
        self.parents = parents


class ActionDAG:
    def __init__(self, sources, sinks):
        self.sources = sources
        self.sinks = sinks

    def get_sinks(self):
        return self.sinks


class Protocol:
    def __init__(self, name, action_dag):
        self.name = name
        self.dag = action_dag

    def get_duration(self):
        terminal_nodes = self.dag.get_sinks()
        longest_duration = 0
        for sink in terminal_nodes:
            duration = self.get_duration_helper(curr_duration=0, next_node=sink)
            if duration > longest_duration:
                longest_duration = duration

        return longest_duration

    def get_duration_helper(self, curr_duration, next_node):
        if not next_node.parents:
            return next_node.duration
        else:
            return max([self.get_duration_helper(curr_duration+next_node.duration, p) for p in next_node.parents])
