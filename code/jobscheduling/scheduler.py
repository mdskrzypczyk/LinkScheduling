import networkx as nx
from abc import abstractmethod
from collections import defaultdict
from math import floor
from queue import PriorityQueue
from jobscheduling.task import get_lcm_for, generate_non_periodic_task_set, generate_non_periodic_budget_task_set, generate_non_periodic_dagtask_set, get_dag_exec_time, PeriodicResourceTask, BudgetTask, PeriodicResourceDAGTask, ResourceTask, ResourceDAGTask
from jobscheduling.log import LSLogger
from copy import copy
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


class PreemptionBudgetScheduler(Scheduler):
    def preprocess_taskset(self, taskset):
        return taskset

    def schedule_tasks(self, taskset):
        original_taskset = taskset
        taskset = self.preprocess_taskset(taskset)
        taskset_copy = [BudgetTask(name=task.name, a=task.a, c=task.c, d=task.d, k=task.k) for task in taskset]
        self.ready_queue = PriorityQueue()
        self.active_queue = []
        self.curr_task = None
        self.schedule = []

        # First sort the taskset by activation time
        self.taskset = list(sorted(taskset, key=lambda task: task.a))

        # Let time evolve and simulate scheduling, start at first task
        self.curr_time = taskset[0].a
        while self.taskset or not self.ready_queue.empty():
            # Get all released tasks into the ready queue
            self.populate_ready_queue()

            # Only need to worry about the active tasks (if any)
            if self.ready_queue.empty():
                # If there is a current task resume it
                if self.curr_task:
                    preempt = self.active_queue and self.active_queue[0][0].k <= 0 and self.curr_task.k > 0
                    if preempt:
                        self.preempt_curr_task()
                        next_active_task, time = self.active_queue.pop(0)
                        self.schedule_until_next_event(next_active_task)
                    else:
                        self.schedule_until_next_event(self.curr_task)

                # No current task, resume an active job
                elif self.active_queue:
                    next_active_task, time = self.active_queue.pop(0)
                    self.schedule_until_next_event(next_active_task)

                elif self.taskset:
                    self.curr_time = self.taskset[0].a

            # We might need to introduce a new task into the active set
            else:
                p, next_ready_task = self.ready_queue.get()
                preempt = True
                active_tasks = [task for task, _ in self.active_queue] if self.curr_task is None else list(sorted([self.curr_task] + [task for task, _ in self.active_queue], key=lambda task: task.k))
                proc_time = next_ready_task.c if not active_tasks else min(active_tasks[0].k, next_ready_task.c)

                # See if the next ready task causes the budget of the active tasks
                preempted_hpwork = 0
                for atask in active_tasks:
                    k_temp = atask.k
                    # Compute the amount of higher priority work in the active queue
                    hpwork = sum([task.c for task, _ in self.active_queue if (task.k <= atask.k and task < atask)]) + proc_time
                    k_temp -= hpwork
                    if k_temp < 0:
                        preempt = False

                    # Check if this task will be higher priority than current task in active queue and add that time
                    if next_ready_task.k <= (atask.k - proc_time) and next_ready_task < atask:
                        k_temp -= (next_ready_task.c - proc_time)
                        if k_temp < 0:
                            preempt = False

                    else:
                        preempted_hpwork += atask.c

                # Check that this new task can tolerate higher priority work in the active queue
                if proc_time < next_ready_task.c:
                    if next_ready_task.k < preempted_hpwork:
                        preempt = False

                # If conditions satisfied preempt the task and run
                if preempt:
                    if self.curr_task:
                        self.preempt_curr_task()
                    self.schedule_until_next_event(next_ready_task)

                # Otherwise run the current task or consume the active queue
                else:
                    self.ready_queue.put((next_ready_task.d, next_ready_task))
                    if self.curr_task:
                        preempt = self.active_queue and self.active_queue[0][0].k <= 0 and self.curr_task.k > 0
                        if preempt:
                            self.preempt_curr_task()
                            next_active_task, time = self.active_queue.pop(0)
                            self.schedule_until_next_event(next_active_task)
                        else:
                            self.schedule_until_next_event(self.curr_task)
                    elif self.active_queue:
                        next_active_task, time = self.active_queue.pop(0)
                        self.schedule_until_next_event(next_active_task)

                    # Nothing to run, fast forward to next release
                    elif self.taskset:
                        self.curr_time = self.taskset[0].a

        self.merge_adjacent_entries()
        self.remove_invalid_entries()

        valid = self.check_feasible(self.schedule, taskset_copy)
        taskset = original_taskset
        return self.schedule, valid

    def check_feasible(self, schedule, taskset):
        # Check validity
        valid = True
        task_starts = {}
        task_ends = {}
        for s, e, task in schedule:
            if not task.name in task_starts.keys():
                task_starts[task.name] = s
            task_ends[task.name] = e

        for task in taskset:
            if task.d < task_ends[task.name]:
                print("Task {} with deadline {} starts at {} and ends at {}".format(task.name, task.d,
                                                                                    task_starts[task.name],
                                                                                    task_ends[task.name]))
                valid = False

            if task_ends[task.name] - task_starts[task.name] > task.c + task.k:
                print("Task {} with budget {} starts at {} and ends at {}".format(task.name, task.k,
                                                                                  task_starts[task.name],
                                                                                  task_ends[task.name]))
                valid = False

        return valid

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

    def remove_invalid_entries(self):
        new_schedule = []
        for s, e, t in self.schedule:
            if s < e:
                new_schedule.append((s, e, t))

        self.schedule = new_schedule
        return new_schedule

    def preempt_curr_task(self):
        task = self.curr_task
        entry_time = self.curr_time
        self.active_queue.append((task, entry_time))
        self.active_queue = list(sorted(self.active_queue, key=lambda t: (t[0].k, t[1])))

    def populate_ready_queue(self):
        while self.taskset and self.taskset[0].a <= self.curr_time:
            task = self.taskset.pop(0)
            self.ready_queue.put((task.d, task))

    def update_active_queue(self, time):
        for task, _ in self.active_queue:
            task.k -= time

    def add_to_schedule(self, task, duration):
        super(PreemptionBudgetScheduler, self).add_to_schedule(task, duration)
        self.update_active_queue(duration)

    def schedule_until_next_event(self, task):
        # Time To Release of next task into ready queue
        ttr = (self.taskset[0].a - self.curr_time) if self.taskset else float('inf')

        # Time To Empty Budget in active queue
        ttb = min(1, self.active_queue[0][0].k) if self.active_queue and task.k > 0 else float('inf')

        # Time to task completion
        ttc = task.c

        # Time until possible to put next ready task into active queue
        ttp = float('inf')

        # Schedule this task to run until the next scheduling decision
        proc_time = min(ttr, ttb, ttc, ttp)
        self.add_to_schedule(task, proc_time)

        # If the amount of time the task is run does not allow it to complete, it will be the current task at the time
        # of the next scheduling decision
        if proc_time < task.c:
            task.c -= proc_time
            self.curr_task = task
        else:
            self.curr_task = None


class PreemptionBudgetSchedulerNew(Scheduler):
    def preprocess_taskset(self, taskset):
        return taskset

    def schedule_tasks(self, taskset):
        original_taskset = taskset
        taskset = self.preprocess_taskset(taskset)
        taskset_copy = [BudgetTask(name=task.name, a=task.a, c=task.c, d=task.d, k=task.k) for task in taskset]
        self.ready_queue = PriorityQueue()
        self.active_queue = []
        self.curr_task = None
        self.schedule = []

        # First sort the taskset by activation time
        self.taskset = list(sorted(taskset, key=lambda task: task.a))

        # Let time evolve and simulate scheduling, start at first task
        self.curr_time = taskset[0].a
        while self.taskset or not self.ready_queue.empty():
            # Get all released tasks into the ready queue
            self.populate_ready_queue()

            # Only need to worry about the active tasks (if any)
            if self.ready_queue.empty() or self.active_queue and self.active_queue[0][0].k <= 0:
                # If there is a current task resume it
                if self.curr_task:
                    preempt = self.active_queue and self.active_queue[0][0].k <= 0 and self.curr_task.k > 0
                    if preempt:
                        self.preempt_curr_task()
                        next_active_task, time = self.active_queue.pop(0)
                        self.schedule_until_next_event(next_active_task)
                    else:
                        self.schedule_until_next_event(self.curr_task)

                # No current task, resume an active job
                elif self.active_queue:
                    next_active_task, time = self.active_queue.pop(0)
                    self.schedule_until_next_event(next_active_task)

                elif self.taskset:
                    self.curr_time = self.taskset[0].a

            # We might need to introduce a new task into the active set
            else:
                p, next_ready_task = self.ready_queue.get()
                preempt = True
                active_tasks = [task for task, _ in self.active_queue] if self.curr_task is None else list(sorted([self.curr_task] + [task for task, _ in self.active_queue], key=lambda task: task.k))

                # First compute the excess budget for each task
                excess_budget = []
                cumulative_comp_time = []
                deadline_slack = []
                comp_time = 0
                for task in active_tasks:
                    excess_budget.append(task.k - comp_time)
                    cumulative_comp_time.append(comp_time)
                    deadline_slack.append(task.d - self.curr_time - task.c - comp_time)
                    comp_time += task.c
                cumulative_comp_time.append(comp_time)

                # Find the earliest place in the active task queue that can tolerate full computation of new task
                first_idx = len(excess_budget)
                for idx in range(len(excess_budget)-1, -1, -1):
                    if excess_budget[idx] < next_ready_task.c or deadline_slack[idx] - next_ready_task.c < 0:
                        break
                    else:
                        first_idx -= 1

                # new task cannot run to completion without violating budgets
                if first_idx != 0:
                    # Otherwise some tasks get violated, see if we can find a place to preempt new task into
                    earliest_idx = first_idx
                    for idx in range(first_idx, len(excess_budget)):
                        if cumulative_comp_time[idx] <= next_ready_task.k:
                            break
                        else:
                            earliest_idx += 1

                    if cumulative_comp_time[earliest_idx - 1] + active_tasks[earliest_idx - 1].c > next_ready_task.k:
                        preempt = False

                    # We want to insert the next_ready_task into this location to respect budgets
                    min_t = max(1, active_tasks[earliest_idx - 1].k - next_ready_task.k)

                    violated_idx = -1
                    max_t = min([task.k for task in active_tasks])
                    for idx in range(earliest_idx - 1, -1, -1):
                        if excess_budget[idx] - min_t < 0 or deadline_slack[idx] - min_t < 0:
                            violated_idx = idx
                        else:
                            max_t = min(max_t, excess_budget[idx], deadline_slack[idx])

                    if violated_idx != -1:
                        preempt = False

                    if max_t - min_t < 0:
                        preempt = False

                    # If conditions satisfied preempt the task and run
                    if preempt:
                        next_ready_task.k = min(next_ready_task.k, next_ready_task.d - next_ready_task.c - cumulative_comp_time[earliest_idx] - self.curr_time)
                        if self.curr_task:
                            self.preempt_curr_task()
                        self.schedule_until_next_event(next_ready_task, max_t)

                    # Otherwise run the current task or consume the active queue
                    else:
                        self.ready_queue.put((next_ready_task.d, next_ready_task))
                        if self.curr_task:
                            preempt = self.active_queue and self.active_queue[0][0].k <= 0 and self.curr_task.k > 0
                            if preempt:
                                self.preempt_curr_task()
                                next_active_task, time = self.active_queue.pop(0)
                                self.schedule_until_next_event(next_active_task)
                            else:
                                self.schedule_until_next_event(self.curr_task)
                        elif self.active_queue:
                            next_active_task, time = self.active_queue.pop(0)
                            self.schedule_until_next_event(next_active_task)

                        # Nothing to run, fast forward to next release
                        elif self.taskset:
                            self.curr_time = self.taskset[0].a

                else:
                    next_ready_task.k = min(next_ready_task.k,
                                            next_ready_task.d - next_ready_task.c - self.curr_time)
                    if self.curr_task:
                        self.preempt_curr_task()
                    self.schedule_until_next_event(next_ready_task)

        self.merge_adjacent_entries()

        valid = self.check_feasible(self.schedule, taskset_copy)
        taskset = original_taskset
        return self.schedule, valid

    def check_feasible(self, schedule, taskset):
        # Check validity
        valid = True
        task_starts = {}
        task_ends = {}
        for s, e, task in schedule:
            if task.name not in task_starts.keys():
                task_starts[task.name] = s
            task_ends[task.name] = e

        for task in taskset:
            if task.d < task_ends[task.name]:
                print("Task {} with deadline {} starts at {} and ends at {}".format(task.name, task.d,
                                                                                  task_starts[task.name],
                                                                                  task_ends[task.name]))
                valid = False

            if task_ends[task.name] - task_starts[task.name] > task.c + task.k:
                print("Task {} with budget {} and comp time {} starts at {} and ends at {}".format(task.name, task.k, task.c,
                                                                                  task_starts[task.name],
                                                                                  task_ends[task.name]))
                valid = False

        return valid

    def get_worst_case_response_times(self, schedule, taskset):
        task_ends = {}
        for _, e, task in schedule:
            task_ends[task.name] = e

        wcrts = defaultdict(int)
        for task in taskset:
            task_name = task.name.split(',')[0]
            wcrts[task_name] = max(wcrts[task_name], task_ends[task.name] - task.a)
        return wcrts

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

    def preempt_curr_task(self):
        task = self.curr_task
        entry_time = self.curr_time
        self.active_queue.append((task, entry_time))
        self.active_queue = list(sorted(self.active_queue, key=lambda t: (t[0].k, t[1])))

    def populate_ready_queue(self):
        while self.taskset and self.taskset[0].a <= self.curr_time:
            task = self.taskset.pop(0)
            self.ready_queue.put((task.d, task))

    def update_active_queue(self, time):
        for task, _ in self.active_queue:
            task.k -= time

    def add_to_schedule(self, task, duration):
        super(PreemptionBudgetSchedulerNew, self).add_to_schedule(task, duration)
        self.update_active_queue(duration)

    def schedule_until_next_event(self, task, ttne=None):
        # Time To Release of next task into ready queue
        ttr = (self.taskset[0].a - self.curr_time) if self.taskset else float('inf')

        # Time to consider next ready task
        ttnr = 1 if not self.ready_queue.empty() else float('inf')

        # Time To Empty Budget in active queue
        ttb = min(1, self.active_queue[0][0].k) if self.active_queue and task.k > 0 else float('inf')

        # Time to task completion
        ttc = task.c

        # Time until possible to put next ready task into active queue
        ttp = float('inf')

        # Schedule this task to run until the next scheduling decision
        proc_time = min(ttr, ttnr, ttb, ttc, ttp)
        if ttne is not None:
            proc_time = min(proc_time, ttne)

        self.add_to_schedule(task, proc_time)

        # If the amount of time the task is run does not allow it to complete, it will be the current task at the time
        # of the next scheduling decision
        if proc_time < task.c:
            task.c -= proc_time
            self.curr_task = task
        else:
            self.curr_task = None


class PeriodicPreemptionBudgetScheduler(PreemptionBudgetScheduler):
    def preprocess_taskset(self, taskset):
        return generate_non_periodic_budget_task_set(taskset)


class PeriodicPreemptionBudgetSchedulerNew(PreemptionBudgetSchedulerNew):
    def preprocess_taskset(self, taskset):
        return generate_non_periodic_budget_task_set(taskset)


class PeriodicEDFScheduler(EDFScheduler):
    def preprocess_taskset(self, taskset):
        return generate_non_periodic_task_set(taskset)


class NPEDFScheduler(Scheduler):
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
                schedule.append((curr_time, curr_time + next_task.c, next_task))
                curr_time += next_task.c

            elif taskset:
                curr_time = taskset[0].a

        # Check validity
        valid = True
        for start, end, task in schedule:
            if task.d < end:
                valid = False
        taskset = original_taskset
        return schedule, valid

    def check_feasibility(self, taskset):
        pass



class MultiResourceNPEDFScheduler(Scheduler):
    def preprocess_taskset(self, taskset):
        return taskset

    def initialize_taskset(self, tasks):
        initial_tasks = []
        for task in tasks:
            task_instance = self.create_new_task_instance(task, 0)
            initial_tasks.append(task_instance)

        return initial_tasks

    def create_new_task_instance(self, periodic_task, instance):
        dag_copy = copy(periodic_task)
        release_offset = dag_copy.a + dag_copy.p*instance
        for subtask in dag_copy.subtasks:
            subtask.a += release_offset
        dag_instance = ResourceDAGTask(name="{}|{}".format(dag_copy.name, instance), a=release_offset,
                                       d=dag_copy.a + dag_copy.p*(instance + 1), tasks=dag_copy.subtasks)
        return dag_instance

    def schedule_tasks(self, taskset):
        original_taskset = taskset
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        logger.debug("Computed hyperperiod {}".format(hyperperiod))
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        last_task_start = defaultdict(int)
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])

        taskset = self.initialize_taskset(taskset)

        resource_schedules = defaultdict(list)
        global_resource_occupations = defaultdict(IntervalTree)

        # First sort the taskset by activation time
        logger.debug("Sorting tasks by earliest deadlines")
        taskset = list(sorted(taskset, key=lambda task: task.a))

        schedule = []
        earliest = 0
        while taskset:
            next_task = taskset.pop(0)
            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            task_resource_occupations = {}
            for resource in next_task.resources:
                task_resource_occupations[resource] = IntervalTree(global_resource_occupations[resource])
                task_resource_occupations[resource].chop(0, max([next_task.a, earliest, last_start]))

            start_time = self.get_start_time(next_task, task_resource_occupations, max([next_task.a, earliest, last_start]))

            # Introduce a new instance into the taskset if necessary
            last_task_start[original_taskname] = start_time
            instance = int(instance)

            if instance < instance_count[original_taskname]:
                periodic_task = taskset_lookup[original_taskname]
                task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                taskset = list(sorted(taskset + [task_instance], key=lambda task: task.a))

            # Add schedule information to resource schedules
            resource_intervals = defaultdict(list)
            for subtask in next_task.subtasks:
                subtask_start = start_time + subtask.a - next_task.a
                subtask_end = subtask_start + subtask.c
                interval = (subtask_start, subtask_end)
                for resource in subtask.resources:
                    resource_intervals[resource].append(interval)
                    resource_schedules[resource].append((subtask_start, subtask_end, subtask))

            # Add the schedule information to the overall schedule
            schedule.append((start_time, start_time + next_task.c, next_task))

            # Update windowed resource schedules
            if taskset:
                min_chop = max(min(list(last_task_start.values())), min(list([t.a for t in taskset])))
                earliest = min_chop
                for resource in next_task.resources:
                    resource_interval_tree = IntervalTree(Interval(begin, end) for begin, end in resource_intervals[resource])
                    global_resource_occupations[resource] |= resource_interval_tree
                    global_resource_occupations[resource].chop(0, min_chop)
                    global_resource_occupations[resource].merge_overlaps(strict=False)

        # Check validity
        valid = True
        for rn, rs in resource_schedules.items():
            for start, end, task in rs:
                if task.d < end:
                    valid = False

        taskset = original_taskset
        return schedule, valid

    def get_start_time(self, task, resource_occupations, earliest):
        subtasks = list(sorted(task.subtasks, key=lambda subtask: subtask.a))
        offset = max(0, earliest - task.a)
        constraining_subtasks = [subtask for subtask in subtasks if self.subtask_constrained(subtask, resource_occupations, offset)]

        # Find the earliest start
        offset = task.a + self.find_earliest_start(constraining_subtasks, resource_occupations)
        while True:
            # See if we can schedule now, otherwise get the minimum number of slots forward before the constrained resource can be scheduled
            scheduleable, step = self.attempt_schedule(constraining_subtasks, offset, resource_occupations)

            if scheduleable:
                return offset

            else:
                # See if we can remove any of the constrained subtasks if we have to move forward by step
                offset += step
                constraining_subtasks = [subtask for subtask in constraining_subtasks if self.subtask_constrained(subtask, resource_occupations, offset)]

    def find_earliest_start(self, subtasks, resource_occupations):
        # Iterate over the tasks and check if their interval overlaps with the resources
        distances_to_free = []
        for subtask in subtasks:
            subtask_start = subtask.a
            subtask_end = subtask_start + subtask.c
            for resource in subtask.resources:
                intervals = sorted(resource_occupations[resource][subtask_start:subtask_end])
                if intervals:
                    overlapping_interval = intervals[0]
                    distances_to_free.append(overlapping_interval.end - subtask_start)

        return max([0] + distances_to_free)

    def attempt_schedule(self, subtasks, start, resource_occupations):
        # Iterate over the tasks and check if their interval overlaps with the resources
        distances_to_free = []
        for subtask in subtasks:
            subtask_start = start + subtask.a
            subtask_end = subtask_start + subtask.c
            for resource in subtask.resources:
                intervals = sorted(resource_occupations[resource][subtask_start:subtask_end])
                if intervals:
                    overlapping_interval = intervals[0]
                    distances_to_free.append(overlapping_interval.end - subtask_start)

        if not distances_to_free:
            return True, 0

        else:
            return False, max(distances_to_free)

    def subtask_constrained(self, subtask, resource_occupations, offset):
        # Check if the start time of this resource is affected by the available intervals
        for resource in subtask.resources:
            rightmost_interval = resource_occupations[resource].end()
            if subtask.a + offset < rightmost_interval:
                return True

        return False


class CEDFScheduler:
    def remove_from_critical_queue(self, cq, task):
        idx = -1
        for i, td in enumerate(cq):
            _, t = td
            if t == task:
                idx = i
                break
        cq.pop(idx)
        return

    def schedule_tasks(self, taskset):
        ready_queue = PriorityQueue()
        critical_queue = list()
        task_to_max_start = {}
        schedule = []

        # First sort the taskset by activation time
        original_taskset = taskset
        taskset = self.preprocess_taskset(taskset)
        taskset = list(sorted(taskset, key=lambda task: task.a))
        for task in taskset:
            critical_queue.append((task.d - task.c, task))
            task_to_max_start[task.name] = task.d - task.c

        critical_queue = list(sorted(critical_queue))

        # Let time evolve and simulate scheduling, start at first task
        curr_time = taskset[0].a
        while taskset or critical_queue:
            while taskset and taskset[0].a <= curr_time:
                task = taskset.pop(0)
                ready_queue.put((task.d, task))

            if not ready_queue.empty():
                priority, next_task = ready_queue.get()
                if curr_time > next_task.a:
                    next_task.a = curr_time

                max_start, next_critical_task = critical_queue[0]
                if curr_time > next_critical_task.a:
                    next_critical_task.a = curr_time

                if next_task.a + next_task.c > max_start and next_task != next_critical_task and next_critical_task.a <= max_start:
                    if (next_task.a + next_task.c) > (next_task.d - next_task.c):
                        # Remove next_task from critical queue
                        max_start = task_to_max_start[next_task.name]
                        critical_queue.remove((max_start, next_task))

                        new_max_start = next_task.a + next_task.c
                        # Reinsert with updated max start time
                        critical_queue.append((new_max_start, next_task))
                        task_to_max_start[next_task.name] = new_max_start
                        critical_queue = list(sorted(critical_queue))

                    next_task.a = next_critical_task.a + next_critical_task.c
                    taskset.append(next_task)
                    taskset = list(sorted(taskset, key=lambda task: task.a))

                else:
                    # Remove next_task from critical queue
                    max_start = task_to_max_start[next_task.name]
                    critical_queue.remove((max_start, next_task))
                    schedule.append((curr_time, curr_time + next_task.c, next_task))
                    curr_time += next_task.c

            elif taskset and ready_queue.empty():
                taskset = list(sorted(taskset, key=lambda task: task.a))
                curr_time = taskset[0].a

        # Check validity
        valid = True
        for start, end, task in schedule:
            if task.d < end:
                # print("Task {} with deadline {} finishes at end time {}".format(task.name, task.d, end))
                valid = False

        return schedule, valid

    def check_feasibility(self, taskset):
        pass

    def preprocess_taskset(self, taskset):
        return taskset


class PeriodicCEDFScheduler(CEDFScheduler):
    def preprocess_taskset(self, taskset):
        return generate_non_periodic_task_set(taskset)


class PeriodicNPEDFScheduler(NPEDFScheduler):
    def preprocess_taskset(self, taskset):
        return generate_non_periodic_task_set(taskset)


class PeriodicMultipleResourceNPEDFScheduler(MultiResourceNPEDFScheduler):
    pass


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


class MultiResourceBlockNPEDFScheduler(Scheduler):
    def preprocess_taskset(self, taskset):
        return taskset

    def initialize_taskset(self, tasks):
        initial_tasks = []
        for task in tasks:
            task_instance = self.create_new_task_instance(task, 0)
            initial_tasks.append(task_instance)

        return initial_tasks

    def create_new_task_instance(self, periodic_task, instance):
        dag_copy = copy(periodic_task)
        release_offset = dag_copy.a + dag_copy.p*instance
        task_instance = ResourceTask(name="{}|{}".format(dag_copy.name, instance), a=release_offset, c=dag_copy.c,
                                     d=dag_copy.a + dag_copy.p*(instance + 1), resources=dag_copy.resources)
        return task_instance

    def schedule_tasks(self, taskset):
        original_taskset = taskset
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        logger.debug("Computed hyperperiod {}".format(hyperperiod))
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        last_task_start = defaultdict(int)
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])

        taskset = self.initialize_taskset(taskset)

        resource_schedules = defaultdict(list)
        global_resource_occupations = defaultdict(IntervalTree)

        # First sort the taskset by activation time
        logger.debug("Sorting tasks by earliest deadlines")
        taskset = list(sorted(taskset, key=lambda task: task.a))

        schedule = []
        earliest = 0
        while taskset:
            next_task = taskset.pop(0)
            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            start_time = self.get_start_time(next_task, global_resource_occupations, max([next_task.a, earliest, last_start]))

            # Introduce a new instance into the taskset if necessary
            last_task_start[original_taskname] = start_time
            instance = int(instance)

            if instance < instance_count[original_taskname]:
                periodic_task = taskset_lookup[original_taskname]
                task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                taskset = list(sorted(taskset + [task_instance], key=lambda task: task.a))

            # Add schedule information to resource schedules
            resource_intervals = defaultdict(list)
            task_start = start_time
            task_end = task_start + next_task.c
            interval = (task_start, task_end)
            for resource in next_task.resources:
                resource_intervals[resource].append(interval)
                resource_schedules[resource].append((task_start, task_end, next_task))

            # Add the schedule information to the overall schedule
            schedule.append((start_time, start_time + next_task.c, next_task))

            # Update windowed resource schedules
            if taskset:
                min_chop = max(min(list(last_task_start.values())), min(list([t.a for t in taskset])))
                earliest = min_chop
                for resource in next_task.resources:
                    resource_interval_tree = IntervalTree(Interval(begin, end) for begin, end in resource_intervals[resource])
                    global_resource_occupations[resource] |= resource_interval_tree
                    global_resource_occupations[resource].chop(0, min_chop)
                    global_resource_occupations[resource].merge_overlaps(strict=False)

        # Check validity
        valid = True
        for rn, rs in resource_schedules.items():
            for start, end, task in rs:
                if task.d < end:
                    valid = False

        taskset = original_taskset
        return schedule, valid

    def get_start_time(self, task, resource_occupations, earliest):
        # Find the earliest start
        offset = self.find_earliest_start(task, resource_occupations, earliest)
        return offset

    def find_earliest_start(self, task, resource_occupations, earliest):
        start = earliest
        distance_to_free = float('inf')
        while distance_to_free != 0:
            sched_interval = Interval(start, start + task.c)

            distance_to_free = 0
            for resource in task.resources:
                intervals = sorted(resource_occupations[resource][sched_interval.begin:sched_interval.end])
                if intervals:
                    overlapping_interval = intervals[0]
                    distance_to_free = max(distance_to_free, overlapping_interval.end - start)

            start += distance_to_free

        return start


class MultipleResourceBlockNPEDFScheduler(Scheduler):
    def schedule_tasks(self, dagset):
        # Convert DAGs into tasks
        tasks = {}
        resources = set()
        for dag_task in dagset:
            block_task = PeriodicResourceDAGTask(name=dag_task.name, tasks=dag_task.subtasks, p=dag_task.p)
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
        scheduler = MultiResourceBlockNPEDFScheduler()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule_tasks(taskset)
            schedules.append((taskset, schedule, valid))

        # Set of schedules is the schedule for each group of resources
        return schedules


class MultipleResourceBlockCEDFScheduler(Scheduler):
    def schedule_tasks(self, dagset):
        # Convert DAGs into tasks
        tasks = {}
        resources = set()
        for dag_task in dagset:
            block_task = PeriodicResourceTask(name=dag_task.name, c=dag_task.c, p=dag_task.p,
                                              resources=dag_task.get_resources())
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
        scheduler = PeriodicCEDFScheduler()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule_tasks(taskset)
            schedules.append((taskset, schedule, valid))

        # Set of schedules is the schedule for each group of resources
        return schedules


class MultipleResourceNonBlockNPEDFScheduler(Scheduler):
    def schedule_tasks(self, dagset):
        # Convert DAGs into tasks
        tasks = {}
        resources = set()
        for dag_task in dagset:
            block_task = PeriodicResourceDAGTask(name=dag_task.name, tasks=dag_task.subtasks, p=dag_task.p)
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
        scheduler = PeriodicMultipleResourceNPEDFScheduler()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule_tasks(taskset)
            schedules.append((taskset, schedule, valid))

        # Set of schedules is the schedule for each group of resources
        return schedules

class MultipleResourceBlockFPPScheduler(Scheduler):
    pass


class MultipleResourceFPPScheduler(Scheduler):
    pass

