import networkx as nx
import pdb
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from math import floor
from queue import PriorityQueue
from task import generate_non_periodic_task_set, generate_non_periodic_budget_task_set, get_dag_exec_time, PeriodicResourceTask, BudgetTask


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


    def schedule(self, taskset):
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
    def schedule(self, taskset):
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
                # print("Task {} with deadline {} finishes at end time {}".format(task.name, task.d, end))
                valid = False

        return schedule, valid

    def check_feasibility(self, taskset):
        pass


class MultiResourceNPEDFScheduler(Scheduler):
    def preprocess_taskset(self, taskset):
        return taskset

    def schedule(self, taskset):
        taskset = self.preprocess_taskset(taskset)
        resource_schedules = defaultdict(list)

        # First sort the taskset by activation time
        taskset = list(sorted(taskset, key=lambda task: task.earliest_deadline()))

        # Let time evolve and simulate scheduling, start at first task
        for next_task in taskset:
            start_time = self.get_start_time(resource_schedules, next_task)
            for subtask in next_task.subtasks:
                subtask_start = start_time + subtask.a
                subtask_end = subtask_start + subtask.c
                for resource_schedule in [resource_schedules[rn] for rn in subtask.resources]:
                    resource_schedule.append((subtask_start, subtask_end, subtask))

        # Check validity
        valid = True
        for rn, rs in resource_schedules.items():
            for start, end, task in rs:
                if task.d < end:
                    # print("Task {} with deadline {} finishes at end time {}".format(task.name, task.d, end))
                    valid = False

        return resource_schedules, valid

    def get_available_slots(self, earliest, latest, schedule):
        slots = []
        for s, e, _ in schedule:
            if earliest < s:
                slots += list(range(earliest, s))
            earliest = e
            if earliest > latest:
                return slots

        if earliest <= latest:
            slots += list(range(earliest, latest + 1))
        return slots

    def intersect_slots(self, slots1, slots2):
        return list(sorted(set(slots1) & set(slots2)))

    def shift_slots(self, slots, amount):
        return [s - amount for s in slots]

    def get_start_time(self, resource_schedules, next_task):
        # Get the available slots of the resources for duration of next_task
        RS = {}
        for rn in next_task.resources:
            rs = resource_schedules[rn]
            earliest = next_task.a
            latest = next_task.d - next_task.c
            rs_slots = self.get_available_slots(earliest, latest, rs)
            RS[rn] = rs_slots

        # Sort subtasks of next_task by start time
        subtasks = sorted(next_task.subtasks, key=lambda subtask: subtask.a)

        subtask_resources = subtasks[0].resources
        subtask_resource_slots = [resource_schedules[resource_name] for resource_name in subtask_resources]
        common_slots = list(sorted(set.intersection(*[set(self.get_available_slots(subtasks[0].a, subtasks[0].d, l))
                                          for l in subtask_resource_slots])))

        starts = []
        for i in range(len(common_slots)):
            if common_slots[i] + subtasks[0].c <= subtasks[0].d and i + (subtasks[0].c - 1) < len(common_slots) and all([common_slots[i+j] - j == common_slots[i] for j in range(1, subtasks[0].c)]):
                starts.append(common_slots[i])

        for subtask in subtasks[1:]:
            resources = subtask.resources
            resource_slots = [resource_schedules[resource_name] for resource_name in resources]
            common_slots = list(sorted(set.intersection(*[set(self.get_available_slots(subtask.a, subtask.d, l))
                                          for l in resource_slots])))
            subtask_starts = []
            for i in range(len(common_slots)):
                if common_slots[i] + subtask.c <= subtask.d and i + (subtask.c - 1) < len(common_slots) and all([common_slots[i + j] - j == common_slots[i] for j in range(1, subtask.c)]):
                    subtask_starts.append(common_slots[i])

            subtask_starts = [s-(subtask.a-subtasks[0].a) for s in subtask_starts]

            starts = set.intersection(set(starts), set(subtask_starts))

        # Return earliest start time
        return min(starts)


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

    def schedule(self, taskset):
        ready_queue = PriorityQueue()
        critical_queue = list()
        schedule = []

        # First sort the taskset by activation time
        taskset = self.preprocess_taskset(taskset)
        taskset = list(sorted(taskset, key=lambda task: task.a))
        for task in taskset:
            critical_queue.append((task.d - task.c, task))

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
                        self.remove_from_critical_queue(critical_queue, next_task)

                        new_max_start = next_task.a + next_task.c
                        # Reinsert with updated max start time
                        critical_queue.append((new_max_start, next_task))
                        critical_queue = list(sorted(critical_queue))

                    next_task.a = next_critical_task.a + next_critical_task.c
                    taskset.append(next_task)
                    taskset = list(sorted(taskset, key=lambda task: task.a))

                else:
                    # Remove next_task from critical queue
                    self.remove_from_critical_queue(critical_queue, next_task)
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


class MultipleResourceOptimalDAGScheduler(Scheduler):
    pass


class MultipleResourceOptimalBlockScheduler(Scheduler):
    def schedule(self, dagset):
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

        sub_graphs = nx.connected_component_subgraphs(G)
        tasksets = []
        for sg in sub_graphs:
            nodes = set(sg.nodes)
            task_names = nodes - resources
            taskset = [tasks[name] for name in task_names]
            tasksets.append(taskset)

        # For each set of tasks use NPEDFScheduler
        scheduler = PeriodicOptimalScheduler()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule(taskset)
            schedules.append(schedule)

        # Set of schedules is the schedule for each group of resources
        return schedules


class MultipleResourceBlockNPEDFScheduler(Scheduler):
    def schedule(self, dagset):
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

        sub_graphs = nx.connected_component_subgraphs(G)
        tasksets = []
        for sg in sub_graphs:
            nodes = set(sg.nodes)
            task_names = nodes - resources
            taskset = [tasks[name] for name in task_names]
            tasksets.append(taskset)

        # For each set of tasks use NPEDFScheduler
        scheduler = PeriodicNPEDFScheduler()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule(taskset)
            schedules.append(schedule)

        # Set of schedules is the schedule for each group of resources
        return schedules


class MultipleResourceBlockCEDFScheduler(Scheduler):
    def schedule(self, dagset):
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

        sub_graphs = nx.connected_component_subgraphs(G)
        tasksets = []
        for sg in sub_graphs:
            nodes = set(sg.nodes)
            task_names = nodes - resources
            taskset = [tasks[name] for name in task_names]
            tasksets.append(taskset)

        # For each set of tasks use NPEDFScheduler
        scheduler = PeriodicCEDFScheduler()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule(taskset)
            schedules.append(schedule)

        # Set of schedules is the schedule for each group of resources
        return schedules


class MultipleResourceBlockFPPScheduler(Scheduler):
    pass


class MultipleResourceNPEDFScheduler(Scheduler):
    pass


class MultipleResourceFPPScheduler(Scheduler):
    pass

