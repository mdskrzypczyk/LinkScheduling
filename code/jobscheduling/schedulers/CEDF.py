import networkx as nx
from queue import PriorityQueue
from jobscheduling.schedulers.scheduler import Scheduler
from jobscheduling.task import PeriodicResourceTask, generate_non_periodic_task_set


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