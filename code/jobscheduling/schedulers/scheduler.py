import networkx as nx
from abc import abstractmethod
from collections import defaultdict
from math import floor, ceil
from queue import PriorityQueue
from jobscheduling.task import get_lcm_for, generate_non_periodic_task_set, generate_non_periodic_budget_task_set, generate_non_periodic_dagtask_set, get_dag_exec_time, PeriodicResourceTask, BudgetTask, PeriodicResourceDAGTask, ResourceTask, ResourceDAGTask
from jobscheduling.log import LSLogger
from intervaltree import IntervalTree, Interval


logger = LSLogger()


def pretty_print_schedule(schedule):
    print([(s, e, (t.name, t.a, t.c, t.d)) for s, e, t in schedule])


def compute_schedule_lateness(schedule):
    return sum([max(0, e - t.d) for s, e, t in schedule])


# Work-conserving, non-preemptive scheduling
def check_wc_np_feasibility(periodic_taskset):
    periodic_taskset = sorted(periodic_taskset, key=lambda task: task.p)

    if not sum([task.c / task.p for task in periodic_taskset]) <= 1:
        return False

    for i in range(len(periodic_taskset)):
        for L in range(periodic_taskset[0].p + 1, periodic_taskset[i].p):
            if L < periodic_taskset[i].c + sum([floor((L-1) / periodic_taskset[j].p) for j in range(i)]):
                return False

    return True


def verify_schedule(tasks, schedule):
    # Construct the occupation intervals of all the resources
    global_resource_intervals = defaultdict(IntervalTree)

    # Iterate over thes chedule
    for start, end, t in schedule:

        # Check that the task's execution period adhere's to it's release and deadline times
        if start < t.a or end > t.d:
            logger.warning("Found task {} ({}, {}) that does not adhere to release/deadline constraints ({}, {})".format(t.name, start, end, t.a, t.d))
            return False

        # Check that the start and end periods align with the tasks runtime
        if end - start != t.c:
            logger.warning("Found task {} that does not have start/end corresponding to duration".format(t.name))
            return False

        # Add the occupation period of this task to all resources
        task_resource_intervals = t.get_resource_intervals()
        offset = start - t.a
        for resource, itree in task_resource_intervals.items():
            offset_itree = IntervalTree([Interval(i.begin + offset, i.end + offset, t) for i in itree])
            for interval in offset_itree:
                if global_resource_intervals[resource].overlap(interval.begin, interval.end):
                    return False
                global_resource_intervals[resource].add(interval)

    return True


class Scheduler:
    def __init__(self):
        self.curr_time = 0
        self.schedule = None
        self.taskset = None

    def add_to_schedule(self, task, duration):
        self.schedule.append((self.curr_time, self.curr_time + duration, task))
        self.curr_time += duration

    @abstractmethod
    def schedule_tasks(self, taskset):
        pass


class OptimalScheduler(Scheduler):
    def rschedule(self, taskset):
        schedule = []

        # First sort the taskset by activation time
        taskset = list(sorted(taskset, key=lambda task: task.d))
        schedule, valid = self.schedule_helper(schedule, taskset)
        return schedule, valid

    def schedule_helper(self, curr_schedule, remaining_tasks):
        if not remaining_tasks:
            return curr_schedule, True

        else:
            next_task = remaining_tasks.pop(0)
            for s in self.get_available_starts(curr_schedule, next_task):
                potential_schedule = list(curr_schedule)
                potential_schedule.append((s, s + next_task.c, next_task))
                sched, valid = self.schedule_helper(potential_schedule, remaining_tasks)
                if valid:
                    return sched, True

        return None, False

    def get_disjoint_tasksets(self, taskset):
        ordered = list(sorted([(task.a, task.d, task.c, task) for task in taskset]))
        disjoint_tasksets = []
        while ordered:
            next_taskset = []
            start, end, dur, next_task = ordered.pop(0)
            next_taskset.append(next_task)
            while ordered and start <= ordered[0][3].a <= end:
                _, _, _, next_task = ordered.pop(0)
                end = max(end, next_task.a)
                next_taskset.append(next_task)
            disjoint_tasksets.append(next_taskset)
        return disjoint_tasksets


    def schedule_tasks(self, taskset):
        schedule = []

        disjoint_tasksets = self.get_disjoint_tasksets(taskset)

        stack = list()

        first_task = taskset[0]
        stack += [(1, [(s, s + first_task.c)]) for s in self.get_available_starts(schedule, first_task)]
        while stack:
            next_task_index, curr_schedule = stack.pop()
            if next_task_index < len(taskset):
                next_task = taskset[next_task_index]
            else:
                return curr_schedule

            available_starts = self.get_available_starts(curr_schedule, next_task)
            if not available_starts:
                continue
            else:
                stack += [(next_task_index + 1, curr_schedule + [(s, s + next_task.c, next_task)]) for s in available_starts]


    def get_available_starts(self, curr_schedule, next_task):
        starting_times = []

        if not curr_schedule:
            return list(range(next_task.a, next_task.d - next_task.c + 1))

        # Check if task can execute before earliest task
        if next_task.a + next_task.c <= curr_schedule[0][0]:
            starting_times += [next_task.a + i for i in range(curr_schedule[0][0] - next_task.a - next_task.c)]

        # Check if task can execute between any tasks
        for td1, td2 in zip(curr_schedule, curr_schedule[1:]):
            s1, e1, t1 = td1
            s2, e2, t2 = td2

            # Check if the task execution period does not lie between these tasks
            if s1 <= next_task.a <= s2:
                starting_times += [e1 + i for i in range(min(s2, next_task.d) - e1 - next_task.c)]
            if s1 >= next_task.d:
                break

        # Check if task can execute after last task
        if next_task.d - next_task.c >= curr_schedule[-1][1]:
            starting_times += [max(curr_schedule[-1][1], next_task.a) + i for i in range(next_task.d - max(curr_schedule[-1][1], next_task.a) - next_task.c + 1)]

        starting_times = list(filter(lambda s: next_task.a <= s <= next_task.d - next_task.c, starting_times))

        return starting_times


class PeriodicOptimalScheduler(OptimalScheduler):
    def schedule_tasks(self, taskset):
        taskset = generate_non_periodic_task_set(taskset)
        disjoint_tasksets = self.get_disjoint_tasksets(taskset)

        schedule = []
        num_scheduled = 0
        for sub_taskset in disjoint_tasksets:
            stack = list()

            first_task = sub_taskset[0]
            starts = self.get_available_starts(schedule, first_task)
            stack += [(1, [(s, s + first_task.c, first_task)]) for s in reversed(starts)]

            curr_schedule = []
            while stack:
                next_task_index, curr_schedule = stack.pop()
                if len(curr_schedule) < len(sub_taskset):
                    next_task = sub_taskset[next_task_index]
                    available_starts = self.get_available_starts(list(sorted(schedule + curr_schedule)), next_task)

                    if available_starts:
                        stack += [(next_task_index + 1, list(sorted(curr_schedule + [(s, s + next_task.c, next_task)])))
                                  for s in reversed(available_starts)]
                else:
                    stack = []

            if len(curr_schedule) != len(sub_taskset):
                return None, False
            else:
                schedule += curr_schedule
                num_scheduled += len(schedule)

        return schedule, True


class EDFScheduler(Scheduler):
    def preprocess_taskset(self, taskset):
        return taskset

    def schedule_tasks(self, taskset):
        original_taskset = taskset
        taskset = self.preprocess_taskset(taskset)
        queue = PriorityQueue()
        schedule = []

        # First sort the taskset by activation time
        taskset = list(sorted(taskset, key=lambda task: task.a))

        # Let time evolve and simulate scheduling, start at first task
        curr_time = taskset[0].a
        while taskset or not queue.empty():
            while taskset and taskset[0].a <= curr_time:
                task = taskset.pop(0)
                queue.put((task.d, task))

            if not queue.empty():
                priority, next_task = queue.get()
                if taskset and curr_time + next_task.c > taskset[0].a:
                    proc_time = taskset[0].a - curr_time
                    next_task.c -= proc_time
                    queue.put((next_task.d, next_task))

                else:
                    proc_time = next_task.c

                schedule.append((curr_time, curr_time + proc_time, next_task))
                curr_time += proc_time

            else:
                curr_time = taskset[0].a

        # Check validity
        valid = True
        for start, end, task in schedule:
            if task.d < end:
                # print("Task {} with deadline {} finishes at end time {}".format(task.name, task.d, end))
                valid = False

        taskset = original_taskset
        return schedule, valid

    def merge_adjacent_entries(self):
        for i in range(len(self.schedule)):
            if i >= len(self.schedule):
                return
            s, e, task = self.schedule[i]
            c = 1
            while i+c < len(self.schedule) and self.schedule[i + c][2].name == task.name:
                e = self.schedule[i+c][1]
                c += 1
            for j in range(1, c):
                self.schedule.pop(i+1)
            self.schedule[i] = (s, e, task)


class PeriodicEDFScheduler(EDFScheduler):
    def preprocess_taskset(self, taskset):
        return generate_non_periodic_task_set(taskset)


class MultipleResourceOptimalBlockScheduler(Scheduler):
    def schedule_tasks(self, dagset):
        # Convert DAGs into tasks
        tasks = {}
        resources = set()
        for dag_task in dagset:
            block_task = PeriodicResourceTask(name=dag_task.name, c=dag_task.c, p=dag_task.p,
                                              resources=dag_task.resources)
            tasks[block_task.name] = block_task
            resources |= block_task.resources

        # Separate tasks based on resource requirements
        G = nx.Graph()
        for r in resources:
            G.add_node(r)
        for block_task in tasks.values():
            G.add_node(block_task.name)
            for r in block_task.resources:
                G.add_edge(block_task.name, r)

        sub_graphs = nx.connected_components(G)
        tasksets = []

        for nodes in sub_graphs:
            task_names = nodes - resources
            taskset = [tasks[name] for name in task_names]
            tasksets.append(taskset)

        # For each set of tasks use NPEDFScheduler
        scheduler = PeriodicOptimalScheduler()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule_tasks(taskset)
            schedules.append(schedule)

        # Set of schedules is the schedule for each group of resources
        return schedules
