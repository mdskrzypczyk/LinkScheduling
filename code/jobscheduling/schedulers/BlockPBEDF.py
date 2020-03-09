import networkx as nx
from collections import defaultdict
from copy import copy
from intervaltree import Interval, IntervalTree
from queue import PriorityQueue
from jobscheduling.log import LSLogger
from jobscheduling.schedulers.scheduler import Scheduler, verify_budget_schedule
from jobscheduling.task import get_lcm_for, generate_non_periodic_budget_task_set, find_dag_task_preemption_points, BudgetTask, BudgetResourceTask, PeriodicBudgetResourceDAGTask


logger = LSLogger()


class UniResourcePreemptionBudgetScheduler(Scheduler):
    def preprocess_taskset(self, taskset):
        return taskset

    def create_new_task_instance(self, periodic_task, instance):
        dag_copy = copy(periodic_task)
        release_offset = dag_copy.a + dag_copy.p * instance
        for subtask in dag_copy.subtasks:
            subtask.a += release_offset
        dag_instance = BudgetResourceTask(name="{}|{}".format(dag_copy.name, instance), a=release_offset, c=dag_copy.c,
                                          d=dag_copy.a + dag_copy.p*(instance + 1), resources=dag_copy.resources,
                                          k=dag_copy.k)
        if dag_instance.d is None:
            import pdb
            pdb.set_trace()
        return dag_instance

    def initialize_taskset(self, tasks):
        initial_tasks = []
        for task in tasks:
            task_instance = self.create_new_task_instance(task, 0)
            initial_tasks.append(task_instance)

        return initial_tasks

    def schedule_tasks(self, taskset, topology):
        original_taskset = taskset
        taskset = self.preprocess_taskset(taskset)
        self.ready_queue = PriorityQueue()
        self.active_queue = []
        self.curr_task = None
        self.schedule = []

        # First sort the taskset by activation time
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        self.taskset_lookup = dict([(t.name, t) for t in original_taskset])
        self.instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])
        self.taskset = self.initialize_taskset(taskset)

        self.taskset = list(sorted(self.taskset, key=lambda task: (task.a, task.d)))

        # Let time evolve and simulate scheduling, start at first task
        self.curr_time = self.taskset[0].a
        while self.taskset or not self.ready_queue.empty() or self.active_queue:
            # Get all released tasks into the ready queue
            self.populate_ready_queue()

            # print("Current ready queue: {}".format([(t[1].name, t[1].c, t[1].k) for t in self.ready_queue.queue]))
            # print("Current active queue: {}".format([(t[0].name, t[0].c, t[0].k) for t in self.active_queue]))
            # print("Current active task: {}".format((self.curr_task.name if self.curr_task else None, self.curr_task.c if self.curr_task else None, self.curr_task.k if self.curr_task else None)))

            # Only need to worry about the active tasks (if any)
            if self.ready_queue.empty() or self.active_queue and self.active_queue[0][0].k <= 0:
                if self.active_queue and self.active_queue[0][0].k < 0:
                    import pdb
                    pdb.set_trace()

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
                    # TODO: Force preemption after max_t
                    if preempt:
                        # next_ready_task.k = min(next_ready_task.k, next_ready_task.d - next_ready_task.c - cumulative_comp_time[earliest_idx] - self.curr_time)
                        if self.curr_task:
                            self.preempt_curr_task()
                        self.schedule_until_next_event(next_ready_task, min_t)
                        if self.curr_task:
                            self.preempt_curr_task()
                            self.curr_task = None

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
                    # next_ready_task.k = min(next_ready_task.k, next_ready_task.d - next_ready_task.c - self.curr_time)
                    if self.curr_task:
                        self.preempt_curr_task()
                    self.schedule_until_next_event(next_ready_task)

        self.merge_adjacent_entries()
        for _, _, t in self.schedule:
            original_taskname, _ = t.name.split('|')
            t.c = self.taskset_lookup[original_taskname].c
            t.k = self.taskset_lookup[original_taskname].k
        valid = verify_budget_schedule(original_taskset, self.schedule)
        taskset = original_taskset
        return [(taskset, self.schedule, valid)]

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
        super(UniResourcePreemptionBudgetScheduler, self).add_to_schedule(task, duration)
        self.update_active_queue(duration)

    def schedule_until_next_event(self, task, ttne=None):
        # Time To Release of next task into ready queue
        ttr = (self.taskset[0].a - self.curr_time) if self.taskset else float('inf')

        # Time to consider next ready task
        ttnr = 1 if not self.ready_queue.empty() else float('inf')

        # Time To Empty Budget in active queue
        ttb = self.active_queue[0][0].k if self.active_queue and task.k > 0 else float('inf')

        # Time to task completion
        ttc = task.c

        # Time until possible to put next ready task into active queue
        ttp = float('inf')

        # Schedule this task to run until the next scheduling decision
        proc_time = min(ttr, ttnr, ttb, ttc, ttp)
        if ttne is not None:
            proc_time = ttne

        # print("Scheduling {} for {}".format(task.name, proc_time))
        self.add_to_schedule(task, proc_time)
        # If the amount of time the task is run does not allow it to complete, it will be the current task at the time
        # of the next scheduling decision
        if proc_time < task.c:
            task.c -= proc_time
            self.curr_task = task
        else:
            original_taskname, instance = task.name.split('|')
            instance = int(instance)
            if instance < self.instance_count[original_taskname]:
                periodic_task = self.taskset_lookup[original_taskname]
                task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                self.taskset = list(sorted(self.taskset + [task_instance], key=lambda task: (task.a, task.d)))
            self.curr_task = None


class MultiResourceBlockPreemptionBudgetScheduler(Scheduler):
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
        release_offset = dag_copy.a + dag_copy.p * instance
        for subtask in dag_copy.subtasks:
            subtask.a += release_offset
        dag_instance = BudgetResourceTask(name="{}|{}".format(dag_copy.name, instance), c=dag_copy.c, a=release_offset,
                                          d=dag_copy.a + dag_copy.p * (instance + 1), resources=dag_copy.resources,
                                          k=periodic_task.k)
        return dag_instance

    def schedule_tasks(self, taskset, topology):
        original_taskset = taskset
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        logger.debug("Computed hyperperiod {}".format(hyperperiod))
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        last_task_start = dict([(t.name, 0) for t in original_taskset])
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])

        taskset = self.initialize_taskset(taskset)

        resource_schedules = defaultdict(list)
        global_resource_occupations = defaultdict(IntervalTree)
        node_resources = topology[1].nodes

        # First sort the taskset by activation time
        logger.debug("Sorting tasks by earliest deadlines")
        taskset = list(sorted(taskset, key=lambda task: task.d))

        schedule = []
        earliest = 0
        while taskset:
            next_task = taskset.pop(0)
            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            start_time = self.get_start_time(next_task, global_resource_occupations, node_resources,
                                             max([next_task.a, earliest, last_start]))
            if start_time + next_task.c > next_task.d:
                return None, False

            # Introduce a new instance into the taskset if necessary
            last_task_start[original_taskname] = start_time
            instance = int(instance)

            if instance < instance_count[original_taskname]:
                periodic_task = taskset_lookup[original_taskname]
                task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                taskset = list(sorted(taskset + [task_instance], key=lambda task: task.d))

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
                    resource_interval_tree = IntervalTree(
                        Interval(begin, end) for begin, end in resource_intervals[resource])
                    global_resource_occupations[resource] |= resource_interval_tree
                    global_resource_occupations[resource].chop(0, min_chop)
                    global_resource_occupations[resource].merge_overlaps(strict=False)

        # Check validity
        valid = verify_budget_schedule(original_taskset, schedule)

        taskset = original_taskset
        return schedule, valid

    def get_start_time(self, task, resource_occupations, node_resources, earliest):
        # Find the earliest start
        offset = self.find_earliest_start(task, resource_occupations, node_resources, earliest)
        return offset

    def find_earliest_start(self, task, resource_occupations, node_resources, earliest):
        start = earliest
        distance_to_free = float('inf')
        while distance_to_free != 0:
            offset = start - task.a
            resource_relations = self.map_task_resources(task, resource_occupations, node_resources, offset)
            task.resources = list(set(resource_relations.values()))

            distance_to_free = 0
            sched_interval = Interval(start, start + task.c)
            for resource in task.resources:
                intervals = sorted(resource_occupations[resource][sched_interval.begin:sched_interval.end])
                if intervals:
                    overlapping_interval = intervals[0]
                    distance_to_free = max(distance_to_free, overlapping_interval.end - start)

            start += distance_to_free

        return start

    def map_task_resources(self, task, resource_occupations, node_resources, offset):
        resource_relations = {}
        for resource in task.resources:
            resource_node, resource_id = resource.split('-')
            resource_type = resource_id[0]
            resource_string = self.get_resource_string(resource)
            if not resource_relations.get(resource_string):
                resource_relations[resource_string] = list(node_resources[resource_node][
                                                               'comm_qs' if resource_type == "C" else "storage_qs"])

        virtual_to_map = {}
        resource_interval_list = [(resource, itree) for resource, itree in
                                  task.get_resource_intervals().items()]
        resource_intervals = list(sorted(resource_interval_list, key=lambda ri: ri[1].begin()))
        for resource, itree in resource_intervals:
            offset_itree = IntervalTree([Interval(i.begin + offset, i.end + offset) for i in itree])
            available_resources = self.sort_map_by_availability(resource, resource_relations,
                                                                resource_occupations,
                                                                offset_itree)
            dist, mapped = available_resources[0]
            virtual_to_map[resource] = mapped
            resource_string = self.get_resource_string(resource)
            resource_relations[resource_string].remove(mapped)

        return virtual_to_map

    def sort_map_by_availability(self, resource, resource_relations, resource_occupations, itree):
        resource_string = self.get_resource_string(resource)
        possible_resources = resource_relations[resource_string]
        available_resources = []
        for pr in possible_resources:
            dist = 0
            for interval in itree:
                intervals = sorted(resource_occupations[pr][interval.begin:interval.end])
                if intervals:
                    overlapping_interval = intervals[0]
                    dist = max(dist, overlapping_interval.end - interval.begin)
            available_resources.append((dist, pr))

        return list(sorted(available_resources))

    def get_resource_string(self, resource):
        resource_node, resource_id = resource.split('-')
        resource_type = resource_id[0]
        return resource_node + resource_type


class MultipleResourceBlockPreemptionBudgetScheduler(Scheduler):
    def schedule_tasks(self, dagset, topology):
        # Convert DAGs into tasks
        tasks = {}
        resources = set()
        for dag_task in dagset:
            block_task = PeriodicBudgetResourceDAGTask(name=dag_task.name, tasks=dag_task.subtasks, p=dag_task.p)
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
        scheduler = MultiResourceBlockPreemptionBudgetScheduler()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule_tasks(taskset, topology)
            schedules.append((taskset, schedule, valid))

        # Set of schedules is the schedule for each group of resources
        return schedules
