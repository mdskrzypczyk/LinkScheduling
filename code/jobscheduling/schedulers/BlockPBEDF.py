from queue import PriorityQueue
from jobscheduling.log import LSLogger
from jobscheduling.schedulers.scheduler import Scheduler, verify_budget_schedule
from jobscheduling.task import get_lcm_for


logger = LSLogger()


class UniResourcePreemptionBudgetScheduler(Scheduler):
    def schedule_tasks(self, taskset, topology=None):
        """
        Main scheduling function for uniprocessor EDF-LBF
        :param taskset: type list
            List of PeriodicTasks to schedule
        :param topology: tuple
            Tuple of networkx.Graphs that represent the communication resources and connectivity graph of the network
        :return: list
            Contains a tuple of (taskset, schedule, valid) where valid indicates if the schedule is valid
        """
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

            # Only need to worry about the active tasks (if any)
            if self.ready_queue.empty() or self.active_queue and self.active_queue[0][0].k <= 0:
                if self.active_queue and self.active_queue[0][0].k < 0:
                    import pdb
                    pdb.set_trace()

                # If there is a current task resume it
                if self.curr_task:
                    preempt = self.active_queue and self.active_queue[0][0].k <= 0 and self.curr_task.k > 0
                    if preempt:
                        next_active_task, time = self.active_queue.pop(0)
                        if self.curr_task != next_active_task:
                            self.preempt_curr_task()

                        if self.curr_time > next_active_task.d - next_active_task.c:
                            return [(taskset, None, False)]

                        self.schedule_until_next_event(next_active_task)
                    else:
                        if self.curr_time > self.curr_task.d - self.curr_task.c:
                            return [(taskset, None, False)]
                        self.schedule_until_next_event(self.curr_task)

                # No current task, resume an active job
                elif self.active_queue:
                    next_active_task, time = self.active_queue.pop(0)
                    if self.curr_time > next_active_task.d - next_active_task.c:
                        return [(taskset, None, False)]
                    self.schedule_until_next_event(next_active_task)

                elif self.taskset:
                    self.curr_time = self.taskset[0].a

            # We might need to introduce a new task into the active set
            else:
                p, next_ready_task = self.ready_queue.get()
                if self.curr_time > next_ready_task.d - next_ready_task.c:
                    return [(taskset, None, False)]
                preempt = True
                active_tasks = [(task, entry_time) for task, entry_time in self.active_queue]
                if self.curr_task is not None:
                    active_tasks.append((self.curr_task, self.curr_time))
                    active_tasks = list(sorted(active_tasks, key=lambda t: (t[0].k, t[1], t[0].name)))

                # First compute the excess budget for each task
                excess_budget = []
                cumulative_comp_time = []
                deadline_slack = []
                comp_time = 0
                for task, _ in active_tasks:
                    excess_budget.append(task.k - comp_time)
                    cumulative_comp_time.append(comp_time)
                    deadline_slack.append(task.d - self.curr_time - task.c - comp_time)
                    comp_time += task.c
                cumulative_comp_time.append(comp_time)

                # Find the earliest place in the active task queue that can tolerate full computation of new task
                first_idx = len(excess_budget)
                for idx in range(len(excess_budget) - 1, -1, -1):
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

                    if cumulative_comp_time[earliest_idx - 1] + active_tasks[earliest_idx - 1][0].c > next_ready_task.k:
                        preempt = False

                    # We want to insert the next_ready_task into this location to respect budgets
                    min_t = max(1, active_tasks[earliest_idx - 1][0].k - next_ready_task.k)

                    violated_idx = -1
                    max_t = min([task.k for task, _ in active_tasks] + deadline_slack)
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
                        if self.curr_task:
                            self.preempt_curr_task()
                            self.curr_task = None
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
                                if self.curr_time > next_active_task.d - next_active_task.c:
                                    return [(taskset, None, False)]
                                self.schedule_until_next_event(next_active_task)
                            else:
                                if self.curr_time > self.curr_task.d - self.curr_task.c:
                                    return [(taskset, None, False)]
                                self.schedule_until_next_event(self.curr_task)
                        elif self.active_queue:
                            next_active_task, time = self.active_queue.pop(0)
                            if self.curr_time > next_active_task.d - next_active_task.c:
                                return [(taskset, None, False)]
                            self.schedule_until_next_event(next_active_task)

                        # Nothing to run, fast forward to next release
                        elif self.taskset:
                            self.curr_time = self.taskset[0].a

                else:
                    if self.curr_task:
                        self.preempt_curr_task()
                    self.schedule_until_next_event(next_ready_task)

        for _, _, t in self.schedule:
            original_taskname, _ = t.name.split('|')
            t.c = self.taskset_lookup[original_taskname].c
            t.k = self.taskset_lookup[original_taskname].k
        valid = verify_budget_schedule(original_taskset, self.schedule)
        taskset = original_taskset
        return [(taskset, self.schedule, valid)]

    def preempt_curr_task(self):
        """
        Preempts the current active task and places it in the active queue
        :return: None
        """
        task = self.curr_task
        entry_time = self.curr_time
        self.active_queue.append((task, entry_time))
        self.active_queue = list(sorted(self.active_queue, key=lambda t: (t[0].k, t[1], t[0].name)))

    def populate_ready_queue(self):
        """
        Populates the ready queue with any tasks that are available, sorted by deadline
        :return: None
        """
        while self.taskset and self.taskset[0].a <= self.curr_time:
            task = self.taskset.pop(0)
            self.ready_queue.put((task.d, task))

    def update_active_queue(self, time):
        """
        Updates the preemption budget of tasks in the active queue
        :param time: type int
            The amount of time to reduce from preemption budgets
        :return: None
        """
        for task, _ in self.active_queue:
            task.k -= time

    def add_to_schedule(self, task, duration):
        """
        Adds a task to the internal schedule and updates the preemption budgets of tasks in the active queue
        :param task: type Task
            The task to add to the schedule
        :param duration: type int
            The amount of time that the task executes for
        :return: Nome
        """
        super(UniResourcePreemptionBudgetScheduler, self).add_to_schedule(task, duration)
        self.update_active_queue(duration)

    def schedule_until_next_event(self, task, ttne=None):
        """
        Schedules a task until the next point we should make a new scheduling decision
        :param task: type Task
            The task to add to the schedule
        :param ttne: type int
            A preset time-til-next-event to run the task for rather than determining from system parameters
        :return: None
        """
        # Time To Release of next task into ready queue
        ttr = (self.taskset[0].a - self.curr_time) if self.taskset else float('inf')

        # Time to consider next ready task
        if not self.ready_queue.empty():
            ttnr = 1
        elif self.taskset:
            ttnr = self.taskset[0].a - self.curr_time
        else:
            ttnr = float('inf')

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


class UniResourceFixedPointPreemptionBudgetScheduler(UniResourcePreemptionBudgetScheduler):
    def schedule_tasks(self, taskset, topology=None):
        """
        Main scheduling function for uniprocessor EDF-LBF with fixed preemption points
        :param taskset: type list
            List of PeriodicTasks to schedule
        :param topology: tuple
            Tuple of networkx.Graphs that represent the communication resources and connectivity graph of the network
        :return: list
            Contains a tuple of (taskset, schedule, valid) where valid indicates if the schedule is valid
        """
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

            # Only need to worry about the active tasks (if any)
            if self.ready_queue.empty() or self.active_queue and self.active_queue[0][0].k <= 0:
                if self.active_queue and self.active_queue[0][0].k < 0:
                    import pdb
                    pdb.set_trace()

                # If there is a current task resume it
                if self.curr_task:
                    preempt = self.active_queue and self.active_queue[0][0].k <= 0 and self.curr_task.k > 0
                    if preempt:
                        next_active_task, time = self.active_queue.pop(0)
                        if self.curr_task != next_active_task:
                            self.preempt_curr_task()

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
                if self.curr_time > next_ready_task.d - next_ready_task.c:
                    return [(taskset, None, False)]
                preempt = True
                active_tasks = [(task, entry_time) for task, entry_time in self.active_queue]
                if self.curr_task is not None:
                    active_tasks.append((self.curr_task, self.curr_time))
                    active_tasks = list(sorted(active_tasks, key=lambda t: (t[0].k, t[1], t[0].name)))

                # First compute the excess budget for each task
                excess_budget = []
                cumulative_comp_time = []
                deadline_slack = []
                comp_time = 0
                for task, _ in active_tasks:
                    excess_budget.append(task.k - comp_time)
                    cumulative_comp_time.append(comp_time)
                    deadline_slack.append(task.d - self.curr_time - task.c - comp_time)
                    comp_time += task.c
                cumulative_comp_time.append(comp_time)

                # Find the earliest place in the active task queue that can tolerate full computation of new task
                first_idx = len(excess_budget)
                for idx in range(len(excess_budget) - 1, -1, -1):
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

                    if cumulative_comp_time[earliest_idx - 1] + active_tasks[earliest_idx - 1][0].c > next_ready_task.k:
                        preempt = False

                    # We want to insert the next_ready_task into this location to respect budgets
                    min_t = max(1, active_tasks[earliest_idx - 1][0].k - next_ready_task.k)

                    violated_idx = -1
                    max_t = min([task.k for task, _ in active_tasks])
                    for idx in range(earliest_idx - 1, -1, -1):
                        if excess_budget[idx] - min_t < 0 or deadline_slack[idx] - min_t < 0:
                            violated_idx = idx
                        else:
                            max_t = min(max_t, excess_budget[idx], deadline_slack[idx])

                    if violated_idx != -1:
                        preempt = False

                    if max_t - min_t < 0:
                        preempt = False

                    original_name_next_ready = next_ready_task.name.split('|')[0]
                    completed_comp_time = self.taskset_lookup[original_name_next_ready].c - next_ready_task.c
                    comp_times = [pp[0][0] - completed_comp_time - next_ready_task.a for pp in
                                  next_ready_task.preemption_points]
                    comp_times = sorted(filter(lambda time: min_t <= time <= max_t, comp_times))

                    if not comp_times:
                        preempt = False

                    # If conditions satisfied preempt the task and run
                    if preempt:
                        if self.curr_task:
                            self.preempt_curr_task()

                        comp_time = comp_times[0]
                        self.schedule_until_next_event(next_ready_task, comp_time)
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
                    if self.curr_task:
                        self.preempt_curr_task()
                    self.schedule_until_next_event(next_ready_task)

        for _, _, t in self.schedule:
            original_taskname, _ = t.name.split('|')
            t.c = self.taskset_lookup[original_taskname].c
            t.k = self.taskset_lookup[original_taskname].k
        valid = verify_budget_schedule(original_taskset, self.schedule)
        taskset = original_taskset
        return [(taskset, self.schedule, valid)]

    def schedule_until_next_event(self, task, ttne=None):
        """
        Schedules a task until the next point we should make a new scheduling decision
        :param task: type Task
            The task to add to the schedule
        :param ttne: type int
            A preset time-til-next-event to run the task for rather than determining from system parameters
        :return: None
        """
        # Time To Empty Budget in active queue
        ttb = self.active_queue[0][0].k if self.active_queue and task.k > 0 else float('inf')

        # Time to task completion
        ttc = task.c

        # Time to next preemption_point
        original_name_next_ready = task.name.split('|')[0]
        completed_comp_time = self.taskset_lookup[original_name_next_ready].c - task.c
        comp_times = [pp[0][0] - completed_comp_time - task.a for pp in
                      task.preemption_points]

        comp_times = sorted(filter(lambda time: time > 0, comp_times))
        max_proc_time = ttb

        if ttc <= max_proc_time:
            proc_time = ttc
        else:
            proc_time = None
            for ct in comp_times:
                if max_proc_time >= ct:
                    proc_time = ct
                else:
                    break

        if proc_time is None:
            import pdb
            pdb.set_trace()

        # Schedule this task to run until the next scheduling decision
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


class UniResourceConsiderateFixedPointPreemptionBudgetScheduler(UniResourcePreemptionBudgetScheduler):
    def schedule_tasks(self, taskset, topology=None):
        """
        Main scheduling function for uniprocessor EDF-LBF with preemption points and resources
        :param taskset: type list
            List of PeriodicTasks to schedule
        :param topology: tuple
            Tuple of networkx.Graphs that represent the communication resources and connectivity graph of the network
        :return: list
            Contains a tuple of (taskset, schedule, valid) where valid indicates if the schedule is valid
        """
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
        while self.taskset or not self.ready_queue.empty() or self.active_queue or self.curr_task:
            # Get all released tasks into the ready queue
            self.populate_ready_queue()

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
                if self.curr_time > next_ready_task.d - next_ready_task.c:
                    return [(taskset, None, False)]
                preempt = True
                active_tasks = [(task, entry_time) for task, entry_time in self.active_queue]
                if self.curr_task is not None:
                    active_tasks.append((self.curr_task, self.curr_time))
                    active_tasks = list(sorted(active_tasks, key=lambda t: (t[0].k, t[1], t[0].name)))

                # First compute the excess budget for each task
                excess_budget = []
                cumulative_comp_time = []
                deadline_slack = []
                comp_time = 0
                for task, _ in active_tasks:
                    excess_budget.append(task.k - comp_time)
                    cumulative_comp_time.append(comp_time)
                    deadline_slack.append(task.d - self.curr_time - task.c - comp_time)
                    comp_time += task.c
                cumulative_comp_time.append(comp_time)

                # Find the earliest place in the active task queue that can tolerate full computation of new task
                first_idx = len(excess_budget)
                for idx in range(len(excess_budget) - 1, -1, -1):
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

                    if cumulative_comp_time[earliest_idx - 1] + active_tasks[earliest_idx - 1][0].c > next_ready_task.k:
                        preempt = False

                    # We want to insert the next_ready_task into this location to respect budgets
                    min_t = max(1, active_tasks[earliest_idx - 1][0].k - next_ready_task.k)

                    violated_idx = -1
                    max_t = min([task.k for task, _ in active_tasks])
                    for idx in range(earliest_idx - 1, -1, -1):
                        if excess_budget[idx] - min_t < 0 or deadline_slack[idx] - min_t < 0:
                            violated_idx = idx
                        else:
                            max_t = min(max_t, excess_budget[idx], deadline_slack[idx])

                    if violated_idx != -1:
                        preempt = False

                    if max_t - min_t < 0:
                        preempt = False

                    original_name_next_ready = next_ready_task.name.split('|')[0]
                    completed_comp_time = self.taskset_lookup[original_name_next_ready].c - next_ready_task.c
                    comp_times = [pp[0][0] - completed_comp_time - next_ready_task.a for pp in
                                  next_ready_task.preemption_points]
                    comp_times = sorted(filter(lambda time: min_t <= time <= max_t, comp_times))

                    if not comp_times:
                        preempt = False

                    else:
                        # Check what resources are currently locked by the active tasks
                        all_locked_resources = []
                        required_resume_resources = []
                        for active_task, entry_time in active_tasks:
                            original_name_active_task = active_task.name.split('|')[0]
                            completed_comp_time = self.taskset_lookup[original_name_active_task].c - active_task.c
                            filtered_points = filter(lambda pp: pp[0][1] - active_task.a == completed_comp_time,
                                                     active_task.preemption_points)
                            current_preemption_point = list(filtered_points)[0]
                            all_locked_resources += current_preemption_point[1]
                            resuming_points = [pp for pp in active_task.preemption_points
                                               if pp[0][0] - active_task.a >= completed_comp_time]
                            required_resume_resources += [r for pp in resuming_points for r in pp[2]]

                        # Check if next_ready_task has the resources it needs and leaves resources unlocked for
                        # resuming tasks
                        next_ready_task_pp = list(filter(lambda pp: pp[0][1] - next_ready_task.a == comp_times[0],
                                                         next_ready_task.preemption_points))[0]
                        next_ready_locked_resources = next_ready_task_pp[1]
                        next_ready_required_resources = next_ready_task_pp[2]

                        if set(next_ready_required_resources) & set(all_locked_resources) or \
                                set(next_ready_locked_resources) & set(required_resume_resources):
                            preempt = False

                    # If conditions satisfied preempt the task and run
                    if preempt:
                        if self.curr_task:
                            self.preempt_curr_task()

                        comp_time = comp_times[0]
                        self.schedule_until_next_event(next_ready_task, comp_time)
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
                    # Check what resources are currently locked by the active tasks
                    all_locked_resources = []
                    for active_task, entry_time in active_tasks:
                        original_name_active_task = active_task.name.split('|')[0]
                        completed_comp_time = self.taskset_lookup[original_name_active_task].c - active_task.c
                        current_preemption_point = list(
                            filter(lambda pp: pp[0][1] - active_task.a == completed_comp_time,
                                   active_task.preemption_points))[0]
                        all_locked_resources += current_preemption_point[1]

                    # Check if next_ready_task has the resources it needs and leaves resources unlocked for resuming
                    # tasks
                    next_ready_task_pp = next_ready_task.preemption_points[-1]
                    next_ready_required_resources = next_ready_task_pp[2]

                    if set(next_ready_required_resources) & set(all_locked_resources):
                        preempt = False

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

        for _, _, t in self.schedule:
            original_taskname, _ = t.name.split('|')
            t.c = self.taskset_lookup[original_taskname].c
            t.k = self.taskset_lookup[original_taskname].k
        valid = verify_budget_schedule(original_taskset, self.schedule)
        taskset = original_taskset
        return [(taskset, self.schedule, valid)]

    def schedule_until_next_event(self, task, ttne=None):
        """
        Schedules a task until the next point we should make a new scheduling decision
        :param task: type Task
            The task to add to the schedule
        :param ttne: type int
            A preset time-til-next-event to run the task for rather than determining from system parameters
        :return: None
        """
        # Time to consider next ready task / release of next task into ready queue
        if not self.ready_queue.empty():
            ttnr = 1
        elif self.taskset:
            ttnr = self.taskset[0].a - self.curr_time
        else:
            ttnr = float('inf')

        # Time To Empty Budget in active queue
        ttb = self.active_queue[0][0].k if self.active_queue and task.k > 0 else float('inf')

        # Time to next preemption_point
        original_name_next_ready = task.name.split('|')[0]
        completed_comp_time = self.taskset_lookup[original_name_next_ready].c - task.c
        comp_times = [pp[0][1] - completed_comp_time - task.a for pp in
                      task.preemption_points]

        comp_times = sorted(filter(lambda time: time > 0, comp_times))
        max_proc_time = ttb

        proc_time = 0
        for ct in comp_times:
            if max_proc_time >= ct:
                proc_time += ct
            if ttnr <= self.curr_time + ct:
                break
            else:
                break

        if proc_time is None or proc_time > max_proc_time:
            import pdb
            pdb.set_trace()

        # Schedule this task to run until the next scheduling decision
        if ttne is not None:
            proc_time = ttne

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
