from queue import PriorityQueue
from jobscheduling.log import LSLogger
from jobscheduling.schedulers.scheduler import Scheduler
from jobscheduling.task import BudgetTask


logger = LSLogger()


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
                active_tasks = [task for task, _ in self.active_queue]
                if self.curr_task is not None:
                    active_tasks.append(self.curr_task)
                    active_tasks = list(sorted(active_tasks, key=lambda t: (t[0].k, t[1], t[0].name)))
                proc_time = next_ready_task.c if not active_tasks else min(active_tasks[0].k, next_ready_task.c)

                # See if the next ready task causes the budget of the active tasks
                preempted_hpwork = 0
                for atask in active_tasks:
                    k_temp = atask.k
                    # Compute the amount of higher priority work in the active queue
                    hpwork = sum([task.c for task, _ in self.active_queue if (task.k <= atask.k and task < atask)])
                    hpwork += proc_time
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

    def merge_adjacent_entries(self):
        for i in range(len(self.schedule)):
            if i >= len(self.schedule):
                return
            s, e, task = self.schedule[i]
            c = 1
            while i + c < len(self.schedule) and self.schedule[i + c][2].name == task.name:
                e = self.schedule[i + c][1]
                c += 1
            for j in range(1, c):
                self.schedule.pop(i + 1)
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
        elif proc_time > task.c:
            import pdb
            pdb.set_trace()
        else:
            self.curr_task = None
