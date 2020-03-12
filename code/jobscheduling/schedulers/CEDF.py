import networkx as nx
from collections import defaultdict
from copy import copy
from intervaltree import Interval, IntervalTree
from queue import PriorityQueue
from jobscheduling.log import LSLogger
from jobscheduling.schedulers.avl_tree import AVLTree
from jobscheduling.schedulers.scheduler import Scheduler, get_lcm_for, verify_schedule
from jobscheduling.task import PeriodicResourceDAGTask, generate_non_periodic_task_set, ResourceTask


logger = LSLogger()


class UniResourceCEDFScheduler:
    def preprocess_taskset(self, taskset):
        return taskset

    def create_new_task_instance(self, periodic_task, instance):
        dag_copy = copy(periodic_task)
        release_offset = dag_copy.a + dag_copy.p * instance
        for subtask in dag_copy.subtasks:
            subtask.a += release_offset
        dag_instance = ResourceTask(name="{}|{}".format(dag_copy.name, instance), c=dag_copy.c, a=release_offset,
                                    d=dag_copy.a + dag_copy.p * (instance + 1), resources=list(dag_copy.resources))
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


class MultiResourceBlockCEDFScheduler(Scheduler):
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

            start_time = self.get_start_time(task_i, global_resource_occupations, node_resources, max([si_min, earliest, last_start]))

            j_starts = [(float('inf'), float('inf'), float('inf'), task_i, ck_i)]
            for resource in taskset_lookup[original_taskname].resources:
                sj_min, sj_max, task_j, ck_j = critical_queue[resource].minimum().data

                if task_j != task_i:
                    j_start = self.get_start_time(task_j, global_resource_occupations, node_resources, max([sj_min, last_start]))
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
                    min_chop = max(min(list(last_task_start.values())), min(list([sj_min for sj_min, _, _, _ in index_structure.values()])))
                    for resource in task_i.resources:
                        resource_interval_tree = IntervalTree(Interval(begin, end) for begin, end in resource_intervals[resource])
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
        for resource in list(sorted(set(task.resources))):
            resource_node, resource_id = resource.split('-')
            resource_type = resource_id[0]
            resource_string = self.get_resource_string(resource)
            if not resource_relations.get(resource_string):
                resource_relations[resource_string] = list(sorted(node_resources[resource_node][
                                                               'comm_qs' if resource_type == "C" else "storage_qs"]))

        virtual_to_map = {}
        resource_interval_list = [(resource, itree) for resource, itree in task.get_resource_intervals().items()]
        resource_intervals = list(sorted(resource_interval_list, key=lambda ri: (ri[1].begin(), ri[0])))
        for resource, itree in resource_intervals:
            offset_itree = IntervalTree([Interval(i.begin + offset, i.end + offset) for i in itree])
            available_resources = self.sort_map_by_availability(resource, resource_relations, resource_occupations,
                                                                offset_itree)
            dist, mapped = available_resources[0]
            virtual_to_map[resource] = mapped
            resource_string = self.get_resource_string(resource)
            resource_relations[resource_string].remove(mapped)

        return virtual_to_map

    def sort_map_by_availability(self, resource, resource_relations, resource_occupations, itree):
        resource_string = self.get_resource_string(resource)
        possible_resources = sorted(resource_relations[resource_string])
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


class MultipleResourceBlockCEDFScheduler(Scheduler):
    def schedule_tasks(self, dagset, topology):
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
        scheduler = MultiResourceBlockCEDFScheduler()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule_tasks(taskset, topology)
            schedules.append((taskset, schedule, valid))

        # Set of schedules is the schedule for each group of resources
        return schedules