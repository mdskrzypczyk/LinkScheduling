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