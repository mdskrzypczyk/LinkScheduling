import networkx as nx
from copy import copy
from queue import PriorityQueue
from jobscheduling.schedulers.avl_tree import AVLTree
from jobscheduling.schedulers.scheduler import Scheduler, get_lcm_for
from jobscheduling.task import PeriodicResourceTask, generate_non_periodic_task_set, ResourceTask


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


class UniResourceCEDFScheduler:
    def preprocess_taskset(self, taskset):
        return taskset

    def create_new_task_instance(self, periodic_task, instance):
        dag_copy = copy(periodic_task)
        release_offset = dag_copy.a + dag_copy.p * instance
        for subtask in dag_copy.subtasks:
            subtask.a += release_offset
        dag_instance = ResourceTask(name="{}|{}".format(dag_copy.name, instance), c=dag_copy.c, a=release_offset,
                                    d=dag_copy.a + dag_copy.p * (instance + 1), resources=dag_copy.resources)
        return dag_instance

    def initialize_taskset(self, tasks):
        initial_tasks = []
        for task in tasks:
            task_instance = self.create_new_task_instance(task, 0)
            initial_tasks.append(task_instance)

        return initial_tasks

    def schedule_tasks(self, taskset, topology):
        ready_queue = PriorityQueue()
        critical_queue = AVLTree()
        index_structure = dict()
        schedule = []

        # First sort the taskset by activation time
        original_taskset = taskset
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])
        taskset = self.initialize_taskset(taskset)

        taskset = list(sorted(taskset, key=lambda task: (task.a, task.d)))
        for task in taskset:
            s_min = task.a
            s_max = task.d - task.c
            original_taskname, instance = task.name.split('|')
            key = (s_max, original_taskname)
            index_structure[original_taskname] = [s_min, s_max, task, (s_max, original_taskname)]
            critical_queue.insert(key=key, data=index_structure[original_taskname])

        # Let time evolve and simulate scheduling, start at first task
        curr_time = taskset[0].a
        while taskset or not critical_queue.is_empty():
            while taskset and taskset[0].a <= curr_time:
                task = taskset.pop(0)
                ready_queue.put((task.d, task))

            if not ready_queue.empty():
                _, task_i = ready_queue.get()
                original_taskname, instance = task_i.name.split('|')
                si_min, si_max, _, ck_i = index_structure[original_taskname]
                sj_min, sj_max, task_j, ck_j = critical_queue.minimum().data

                if si_min + task_i.c > sj_max and task_i != task_j and sj_min <= sj_max:
                    if si_min + task_i.c > si_max:
                        # Remove task_i from critical queue
                        critical_queue.delete(key=ck_i)

                        # Reinsert
                        ck_i = ((si_min + task_i.c), original_taskname)
                        index_structure[original_taskname] = [si_min, si_max, task_i, ck_i]
                        critical_queue.insert(key=ck_i, data=index_structure[original_taskname], note=si_max)

                    index_structure[original_taskname][0] = sj_min + task_j.c
                    task_i.a = sj_min + task_j.c
                    taskset.append(task_i)
                    taskset = list(sorted(taskset, key=lambda task: (task.a, task.d)))

                else:
                    # Remove next_task from critical queue
                    critical_queue.delete(ck_i)
                    schedule.append((curr_time, curr_time + task_i.c, task_i))
                    instance = int(instance)
                    if instance < instance_count[original_taskname]:
                        periodic_task = taskset_lookup[original_taskname]
                        task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                        taskset = list(sorted(taskset + [task_instance], key=lambda task: (task.a, task.d)))
                        s_min = task_instance.a
                        s_max = task_instance.d - task_instance.c
                        key = (s_max, original_taskname)
                        index_structure[original_taskname] = [s_min, s_max, task_instance, key]
                        critical_queue.insert(key=key, data=index_structure[original_taskname])

                    curr_time += task_i.c

            elif taskset and ready_queue.empty():
                repeat = 0
                taskset = list(sorted(taskset, key=lambda task: (task.a, task.d)))
                curr_time = taskset[0].a

        # Check validity
        valid = True
        for start, end, task in schedule:
            if task.d < end:
                # print("Task {} with deadline {} finishes at end time {}".format(task.name, task.d, end))
                valid = False

        return [(original_taskset, schedule, valid)]

    def check_feasibility(self, taskset):
        pass


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