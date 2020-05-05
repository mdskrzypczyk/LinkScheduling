from collections import defaultdict
from intervaltree import Interval, IntervalTree
from queue import PriorityQueue
from jobscheduling.log import LSLogger
from jobscheduling.schedulers.avl_tree import AVLTree
from jobscheduling.schedulers.scheduler import Scheduler, BaseMultipleResourceScheduler, get_lcm_for, verify_schedule


logger = LSLogger()


class UniResourceCEDFScheduler(Scheduler):
    def schedule_tasks(self, taskset, topology=None):
        """
        Main scheduling function for uniprocessor CEDF
        :param taskset: type list
            List of PeriodicTasks to schedule
        :param topology: tuple
            Tuple of networkx.Graphs that represent the communication resources and connectivity graph of the network
        :return: list
            Contains a tuple of (taskset, schedule, valid) where valid indicates if the schedule is valid
        """
        ready_queue = PriorityQueue()
        critical_queue = AVLTree()
        index_structure = dict()
        schedule = []

        # First sort the taskset by activation time
        original_taskset = taskset
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])
        taskset = self.initialize_taskset(original_taskset)

        taskset = list(sorted(taskset, key=lambda task: (task.d, task.a)))
        for task in taskset:
            s_min = task.a
            s_max = task.d - task.c
            original_taskname, instance = task.name.split('|')
            key = (s_max, original_taskname)
            index_structure[original_taskname] = [s_min, s_max, task, (s_max, original_taskname)]
            critical_queue.insert(key=key, data=index_structure[original_taskname])

        # Let time evolve and simulate scheduling, start at first task
        curr_time = taskset[0].a
        while taskset or not critical_queue.is_empty() or not ready_queue.empty():
            while taskset and taskset[0].a <= curr_time:
                task = taskset.pop(0)
                ready_queue.put((task.d, task))

            if not ready_queue.empty():
                _, task_i = ready_queue.get()
                original_taskname, instance = task_i.name.split('|')
                index_structure[original_taskname][0] = curr_time
                si_min, si_max, _, ck_i = index_structure[original_taskname]
                sj_min, sj_max, task_j, ck_j = critical_queue.minimum().data
                original_taskname_j = task_j.name.split('|')[0]
                index_structure[original_taskname_j][0] = curr_time
                sj_min = curr_time

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
                    if curr_time + task_i.c > task_i.d:
                        return [(None, None, False)]
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
                taskset = list(sorted(taskset, key=lambda task: (task.a, task.d)))
                curr_time = taskset[0].a

        # Check validity
        valid = True
        for start, end, task in schedule:
            if task.d < end:
                valid = False

        return [(original_taskset, schedule, valid)]


class MultiResourceBlockCEDFScheduler(Scheduler):
    def schedule_tasks(self, taskset, topology):
        """
        Main scheduling function for RCPSP CEDF
        :param taskset: type list
            List of PeriodicTasks to schedule
        :param topology: tuple
            Tuple of networkx.Graphs that represent the communication resources and connectivity graph of the network
        :return: list
            Contains a tuple of (taskset, schedule, valid) where valid indicates if the schedule is valid
        """
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

        critical_queue = defaultdict(AVLTree)
        index_structure = defaultdict(list)
        taskset = list(sorted(taskset, key=lambda task: (task.d, task.a, task.name)))
        for task in taskset:
            s_min = task.a
            s_max = task.d - task.c
            original_taskname, instance = task.name.split('|')
            key = (s_max, original_taskname)
            index_structure[original_taskname] = [s_min, s_max, task, (s_max, original_taskname)]
            for resource in taskset_lookup[original_taskname].resources:
                critical_queue[resource].insert(key=key, data=index_structure[original_taskname])

        schedule = []
        earliest = 0
        while taskset:
            task_i = taskset.pop(0)
            original_taskname, instance = task_i.name.split('|')
            si_min, si_max, _, ck_i = index_structure[original_taskname]
            last_start = last_task_start[original_taskname]

            start_time = self.get_start_time(task_i, global_resource_occupations, node_resources,
                                             max([si_min, earliest, last_start]))

            j_starts = [(float('inf'), float('inf'), float('inf'), task_i, ck_i)]
            for resource in taskset_lookup[original_taskname].resources:
                sj_min, sj_max, task_j, ck_j = critical_queue[resource].minimum().data

                if task_j != task_i:
                    j_start = self.get_start_time(task_j, global_resource_occupations, node_resources,
                                                  max([sj_min, last_start]))
                    taskname_j, instance = task_j.name.split('|')
                    index_structure[taskname_j][0] = j_start
                    sj_min = j_start
                    j_starts.append((j_start, sj_min, sj_max, task_j, ck_j))

            j_start, sj_min, sj_max, task_j, ck_j = sorted(j_starts)[0]

            if start_time + task_i.c > sj_max and task_i != task_j and j_start <= sj_max:
                if si_min + task_i.c > si_max:
                    for resource in taskset_lookup[original_taskname].resources:
                        # Remove task_i from critical queue
                        critical_queue[resource].delete(key=ck_i)

                    ck_i = ((si_min + task_i.c), original_taskname)
                    index_structure[original_taskname] = [si_min, si_max, task_i, ck_i]
                    for resource in taskset_lookup[original_taskname].resources:
                        # Reinsert
                        critical_queue[resource].insert(key=ck_i, data=index_structure[original_taskname], note=si_max)

                index_structure[original_taskname][0] = sj_min + task_j.c

                task_i.a = sj_min + task_j.c
                taskset.append(task_i)
                taskset = list(sorted(taskset, key=lambda task: (task.a, task.d)))

            else:
                if start_time + task_i.c > task_i.d:
                    return None, False

                # Remove next_task from critical queue
                for resource in taskset_lookup[original_taskname].resources:
                    critical_queue[resource].delete(ck_i)

                # Add the schedule information to the overall schedule
                schedule.append((start_time, start_time + task_i.c, task_i))

                # Introduce a new instance into the taskset if necessary
                last_task_start[original_taskname] = start_time
                instance = int(instance)

                if instance < instance_count[original_taskname]:
                    periodic_task = taskset_lookup[original_taskname]
                    task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                    taskset = list(sorted(taskset + [task_instance], key=lambda task: (task.d, task.a, task.name)))
                    s_min = task_instance.a
                    s_max = task_instance.d - task_instance.c
                    key = (s_max, original_taskname)
                    index_structure[original_taskname] = [s_min, s_max, task_instance, key]
                    for resource in taskset_lookup[original_taskname].resources:
                        critical_queue[resource].insert(key=key, data=index_structure[original_taskname])

                # Add schedule information to resource schedules
                resource_intervals = defaultdict(list)
                task_start = start_time
                task_end = task_start + task_i.c
                interval = (task_start, task_end)
                for resource in task_i.resources:
                    resource_intervals[resource].append(interval)
                    resource_schedules[resource].append((task_start, task_end, task_i))

                # Update windowed resource schedules
                if taskset:
                    min_chop = max(min(list(last_task_start.values())), min(list([sj_min for sj_min, _, _, _ in
                                                                                  index_structure.values()])))
                    for resource in task_i.resources:
                        resource_interval_tree = IntervalTree(Interval(begin, end) for begin, end in
                                                              resource_intervals[resource])
                        if global_resource_occupations[resource] & resource_interval_tree:
                            import pdb
                            pdb.set_trace()
                        global_resource_occupations[resource] |= resource_interval_tree
                        global_resource_occupations[resource].chop(0, min_chop)
                        global_resource_occupations[resource].merge_overlaps(strict=False)

        # Check validity
        valid = verify_schedule(original_taskset, schedule)

        taskset = original_taskset
        return schedule, valid

    def get_start_time(self, task, resource_occupations, node_resources, earliest):
        """
        Obtains the start time for a task
        :param task: type DAGTask
            The task to obtain the start time for
        :param resource_occupations: type dict(IntervalTree)
            A dictionary of resource identifiers to interval trees representing resource occupations
        :param node_resources: type dict
            A dictionary of node to resource identifiers held by a node
        :param earliest: type int
            The earliest point in time the task is permitted to start
        :return: type int
            The earliest time the incoming task may start
        """
        # Find the earliest start
        offset = self.find_earliest_start(task, resource_occupations, node_resources, earliest)
        return offset

    def find_earliest_start(self, task, resource_occupations, node_resources, earliest):
        """
        Finds the start time for a task
        :param task: type DAGTask
            The task to obtain the start time for
        :param resource_occupations: type dict(IntervalTree)
            A dictionary of resource identifiers to interval trees representing resource occupations
        :param node_resources: type dict
            A dictionary of node to resource identifiers held by a node
        :param earliest: type int
            The earliest point in time the task is permitted to start
        :return: type int
            The earliest time the incoming task may start
        """
        start = earliest
        distance_to_free = float('inf')
        while distance_to_free != 0:
            distance_to_free = 0
            sched_interval = Interval(start, start + task.c)
            for resource in task.resources:
                intervals = sorted(resource_occupations[resource][sched_interval.begin:sched_interval.end])
                if intervals:
                    overlapping_interval = intervals[0]
                    distance_to_free = max(distance_to_free, overlapping_interval.end - start)

            start += distance_to_free

        return start


class MultipleResourceBlockCEDFScheduler(BaseMultipleResourceScheduler):
    internal_scheduler_class = MultiResourceBlockCEDFScheduler
