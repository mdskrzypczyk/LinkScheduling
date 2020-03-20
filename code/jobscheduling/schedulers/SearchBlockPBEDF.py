import networkx as nx
from collections import defaultdict
from copy import copy
from intervaltree import Interval, IntervalTree
from queue import PriorityQueue
from jobscheduling.log import LSLogger
from jobscheduling.schedulers.scheduler import Scheduler, verify_budget_schedule, verify_segmented_budget_schedule
from jobscheduling.task import get_lcm_for, generate_non_periodic_budget_task_set, find_dag_task_preemption_points, BudgetTask, BudgetResourceTask, BudgetResourceDAGTask, PeriodicBudgetResourceDAGTask, ResourceTask, ResourceDAGTask


logger = LSLogger()


class CommonScheduler:
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

        dag_instance = BudgetResourceDAGTask(name="{}|{}".format(dag_copy.name, instance), a=release_offset,
                                             d=dag_copy.a + dag_copy.p * (instance + 1), tasks=dag_copy.subtasks,
                                             k=periodic_task.k)

        return dag_instance

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

    def remap_task_resources(self, task, offset, resource_occupations, node_resources):
        resource_relations = self.map_task_resources(task, resource_occupations, node_resources, offset)
        task.resources = list(sorted(set(resource_relations.values())))
        for subtask in task.subtasks:
            new_resources = []
            new_locked_resources = []
            for resource in subtask.resources:
                new_resources.append(resource_relations[resource])
            for resource in subtask.locked_resources:
                new_locked_resources.append(resource_relations[resource])
            subtask.resources = new_resources
            subtask.locked_resources = new_locked_resources

        return task

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

    def update_resource_occupations(self, resource_occupations, resource_intervals):
        # Update windowed resource schedules
        for resource in resource_intervals.keys():
            resource_interval_tree = IntervalTree(
                Interval(begin, end) for begin, end, _ in resource_intervals[resource])
            if any([resource_occupations[resource].overlap(begin, end) for begin, end, _ in
                    resource_intervals[resource]]):
                import pdb
                pdb.set_trace()
            resource_occupations[resource] |= resource_interval_tree
            resource_occupations[resource].merge_overlaps(strict=False)

    def remove_useless_resource_occupations(self, resource_occupations, resources, chop):
        for resource in resources:
            resource_occupations[resource].chop(0, chop)

    def schedule_tasks(self, taskset, topology):
        original_taskset = taskset
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        logger.debug("Computed hyperperiod {}".format(hyperperiod))
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        last_task_start = dict([(t.name, 0) for t in original_taskset])
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])

        taskset = self.initialize_taskset(taskset)

        global_resource_occupations = defaultdict(IntervalTree)
        node_resources = topology[1].nodes

        # First sort the taskset by activation time
        logger.debug("Sorting tasks by earliest deadlines")
        taskset = list(sorted(taskset, key=lambda task: (task.d, task.a, task.name)))

        schedule = []
        earliest = 0
        while taskset:
            next_task = taskset.pop(0)
            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            preemption_point_intervals = self.get_start_time(next_task, global_resource_occupations, node_resources,
                                                             max([next_task.a, earliest, last_start]))

            # Check if time violated deadline
            last_pp_interval, last_pp_tasks = preemption_point_intervals[-1]
            last_pp_start, last_pp_end = last_pp_interval
            if last_pp_end > next_task.d:
                return None, False

            # Record start time
            first_interval, _ = preemption_point_intervals[0]
            start_time = first_interval[0]
            last_task_start[original_taskname] = start_time

            # Introduce a new instance into the taskset if necessary
            instance = int(instance)
            if instance < instance_count[original_taskname]:
                periodic_task = taskset_lookup[original_taskname]
                task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                taskset = list(sorted(taskset + [task_instance], key=lambda task: (task.d, task.a, task.name)))

            # Add schedule information to resource schedules
            resource_intervals = self.extract_resource_intervals_from_preemption_point_intervals(preemption_point_intervals)
            self.schedule_preemption_point_intervals(schedule, preemption_point_intervals)

            # Update windowed resource schedules
            if taskset:
                self.update_resource_occupations(global_resource_occupations, resource_intervals)
                min_chop = max(min(list(last_task_start.values())), min(list([t.a for t in taskset])))
                self.remove_useless_resource_occupations(global_resource_occupations, resource_intervals.keys(), min_chop)

        # Check validity
        valid = self.verify_schedule(original_taskset, schedule)

        taskset = original_taskset
        return schedule, valid


class MultiResourceInconsiderateFixedPointBlockPreemptionBudgetScheduler(CommonScheduler):
    def verify_schedule(self, taskset, schedule):
        return verify_budget_schedule(taskset, schedule)

    def extract_resource_intervals_from_preemption_point_intervals(self, preemption_point_intervals):
        # Add schedule information to resource schedules
        resource_intervals = defaultdict(IntervalTree)
        for segment_interval, segment_task in preemption_point_intervals:
            interval = Interval(begin=segment_interval[0], end=segment_interval[1])
            for resource in segment_task.resources:
                resource_intervals[resource].add(interval)

        return resource_intervals

    def schedule_preemption_point_intervals(self, schedule, preemption_point_intervals):
        for segment_interval, segment_task in preemption_point_intervals:
            segment_start, segment_end = segment_interval
            schedule.append((segment_start, segment_end, segment_task))

    def get_segment_tasks(self, task):
        segment_tasks = []
        segment_info = find_dag_task_preemption_points(task)
        comp_time = 0
        for i, segment in enumerate(segment_info):
            segment_times, segment_locked_resources, segment_resources, segment_subtasks = segment
            segment_start_offset, segment_end_offset = segment_times
            segment_duration = segment_end_offset - segment_start_offset
            segment_task = ResourceTask(name="{}|{}".format(task.name, i), c=segment_duration,
                                        a=task.a + comp_time, d=task.d - task.c + comp_time + segment_duration,
                                        resources=task.resources,
                                        locked_resources=task.resources)
            segment_tasks.append(segment_task)
            comp_time += segment_duration

        return segment_tasks

    def get_start_time(self, task, resource_occupations, node_resources, earliest):
        segment_earliest = earliest
        # Find the earliest start

        while True:
            offset = segment_earliest - task.a
            task = self.remap_task_resources(task, offset, resource_occupations, node_resources)
            segment_tasks = self.get_segment_tasks(task)

            segment_intervals = []
            attempt_earliest = segment_earliest
            comp_time = 0
            for segment_task in segment_tasks:
                # Find the earliest start
                segment_start = self.find_earliest_start(segment_task, resource_occupations, attempt_earliest)
                if not segment_intervals and segment_start != segment_earliest:
                    segment_earliest = segment_start
                    break

                attempt_earliest = segment_start + segment_task.c
                comp_time += segment_task.c
                segment_intervals.append((segment_start, segment_start + segment_task.c))
                if segment_start + segment_task.c - segment_intervals[0][0] > comp_time + task.k:
                    segment_earliest = segment_start - comp_time + segment_task.c - task.k
                    break

                elif len(segment_intervals) == len(segment_tasks):
                    start_times = list(zip(segment_intervals, [segment_task for segment_task in segment_tasks]))
                    return start_times

    def find_earliest_start(self, task, resource_occupations, earliest):
        start = earliest
        distance_to_free = 1
        while distance_to_free:
            distance_to_free = 0
            sched_interval = Interval(start, start + task.c)
            for resource in task.resources:
                intervals = sorted(resource_occupations[resource][sched_interval.begin:sched_interval.end])
                if intervals:
                    overlapping_interval = intervals[0]
                    distance_to_free = max(distance_to_free, overlapping_interval.end - start)

            start += distance_to_free

        return start


class MultiResourceInconsiderateFixedPointSegmentBlockPreemptionBudgetScheduler(MultiResourceInconsiderateFixedPointBlockPreemptionBudgetScheduler):
    def verify_schedule(self, taskset, schedule):
        return verify_segmented_budget_schedule(taskset, schedule)

    def extract_resource_intervals_from_preemption_point_intervals(self, preemption_point_intervals):
        # Add schedule information to resource schedules
        resource_intervals = defaultdict(IntervalTree)
        for segment_interval, segment_task in preemption_point_intervals:
            interval = Interval(begin=segment_interval[0], end=segment_interval[1], data=segment_task)
            for resource in segment_task.resources:
                resource_intervals[resource].add(interval)

        return resource_intervals

    def get_start_time(self, task, resource_occupations, node_resources, earliest):
        segment_earliest = earliest
        # Find the earliest start
        while True:
            offset = segment_earliest - task.a
            task = self.remap_task_resources(task, offset, resource_occupations, node_resources)
            segment_tasks = self.get_segment_tasks(task)

            segment_intervals = []
            for segment_task in segment_tasks:
                # Find the earliest start
                segment_start = self.find_earliest_start(segment_task, resource_occupations, segment_earliest)
                if not segment_intervals and segment_start != segment_earliest:
                    segment_earliest = segment_start
                    break

                segment_earliest = segment_start + segment_task.c
                segment_intervals.append((segment_start, segment_start + segment_task.c))
                if segment_start + segment_task.c - segment_intervals[0][0] > task.c + task.k:
                    segment_earliest = segment_start - sum(
                        [interval[1] - interval[0] for interval in segment_intervals[:-1]]) - task.k

                    break

                elif len(segment_intervals) == len(segment_tasks):
                    start_times = list(zip(segment_intervals, [segment_task for segment_task in segment_tasks]))
                    return start_times

    def get_segment_tasks(self, task):
        segment_tasks = []
        segment_info = find_dag_task_preemption_points(task)
        comp_time = 0
        for i, segment in enumerate(segment_info):
            segment_times, segment_locked_resources, segment_resources, segment_subtasks = segment
            segment_start_offset, segment_end_offset = segment_times
            segment_duration = segment_end_offset - segment_start_offset
            segment_task = ResourceTask(name="{}|{}".format(task.name, i), c=segment_duration,
                                        a=task.a + comp_time, d=task.d - task.c + comp_time + segment_duration,
                                        resources=segment_resources,
                                        locked_resources=segment_locked_resources)
            segment_tasks.append(segment_task)
            comp_time += segment_duration

        return segment_tasks

    def find_earliest_start(self, task, resource_occupations, earliest):
        start = earliest
        distance_to_free = 1
        while distance_to_free:
            distance_to_free = 0
            sched_interval = Interval(start, start + task.c)
            for resource in task.resources:
                intervals = sorted(resource_occupations[resource][sched_interval.begin:sched_interval.end])
                if intervals:
                    overlapping_interval = intervals[0]
                    distance_to_free = max(distance_to_free, overlapping_interval.end - start)

            start += distance_to_free

        return start


class MultiResourceInconsiderateFixedPointSegmentPreemptionBudgetScheduler(MultiResourceInconsiderateFixedPointSegmentBlockPreemptionBudgetScheduler):
    def extract_resource_intervals_from_preemption_point_intervals(self, preemption_point_intervals):
        resource_intervals = defaultdict(IntervalTree)
        for segment_interval, segment_task in preemption_point_intervals:
            segment_start, segment_end = segment_interval
            segment_interval_list = [(resource, itree) for resource, itree in
                                     segment_task.get_resource_intervals().items()]
            segment_intervals = list(sorted(segment_interval_list, key=lambda ri: ri[1].begin()))
            for resource, itree in segment_intervals:
                offset_itree = IntervalTree(
                    [Interval(i.begin + segment_start - segment_task.a, i.end + segment_start - segment_task.a) for i in
                     itree])
                resource_intervals[resource] |= offset_itree

        return resource_intervals

    def get_start_time(self, task, resource_occupations, node_resources, earliest):
        segment_earliest = earliest
        # Find the earliest start
        while True:
            offset = segment_earliest - task.a
            task = self.remap_task_resources(task, offset, resource_occupations, node_resources)
            segment_tasks = self.get_segment_tasks(task)

            segment_intervals = []
            for segment_task in segment_tasks:
                # Find the earliest start
                segment_start = self.find_earliest_start(segment_task, resource_occupations, segment_earliest)
                if not segment_intervals and segment_start != segment_earliest:
                    segment_earliest = segment_start
                    break

                segment_earliest = segment_start + segment_task.c
                segment_intervals.append((segment_start, segment_start + segment_task.c))
                if segment_start + segment_task.c - segment_intervals[0][0] > task.c + task.k:
                    segment_earliest = segment_start - sum(
                        [interval[1] - interval[0] for interval in segment_intervals[:-1]]) - task.k

                    break

                elif len(segment_intervals) == len(segment_tasks):
                    start_times = list(zip(segment_intervals, [segment_task for segment_task in segment_tasks]))
                    return start_times

    def get_segment_tasks(self, task):
        segment_tasks = []
        segment_info = find_dag_task_preemption_points(task)
        comp_time = 0
        for i, segment in enumerate(segment_info):
            segment_times, segment_locked_resources, segment_resources, segment_subtasks = segment
            segment_start_offset, segment_end_offset = segment_times
            segment_duration = segment_end_offset - segment_start_offset
            segment_task = ResourceDAGTask(name="{}|{}".format(task.name, i), a=segment_start_offset,
                                           d=max([segment_subtask.d for segment_subtask in segment_subtasks]),
                                           tasks=segment_subtasks)
            segment_tasks.append(segment_task)
            comp_time += segment_duration

        return segment_tasks

    def find_earliest_start(self, task, resource_occupations, earliest):
        start = earliest
        distance_to_free = 1
        while distance_to_free:
            distance_to_free = 0
            resource_interval_list = [(resource, itree) for resource, itree in task.get_resource_intervals().items()]
            resource_intervals = list(sorted(resource_interval_list, key=lambda ri: ri[1].begin()))
            offset = start - task.a
            for resource, itree in resource_intervals:
                for interval in itree:
                    intervals = sorted(resource_occupations[resource][interval.begin + offset:interval.end + offset])
                    if intervals:
                        overlapping_interval = intervals[0]
                        distance_to_free = max(distance_to_free, overlapping_interval.end - (interval.begin + offset))

            start += distance_to_free

        return start


class MultiResourceConsiderateFixedPointBlockPreemptionBudgetScheduler(CommonScheduler):
    def schedule_tasks(self, taskset, topology):
        original_taskset = taskset
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        logger.debug("Computed hyperperiod {}".format(hyperperiod))
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        last_task_start = dict([(t.name, 0) for t in original_taskset])
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])

        taskset = self.initialize_taskset(taskset)

        global_resource_occupations = defaultdict(IntervalTree)
        node_resources = topology[1].nodes

        # First sort the taskset by activation time
        logger.debug("Sorting tasks by earliest deadlines")
        taskset = list(sorted(taskset, key=lambda task: (task.d, task.a, task.name)))

        schedule = []
        earliest = 0
        while taskset:
            next_task = taskset.pop(0)
            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            preemption_point_intervals = self.get_start_time(next_task, global_resource_occupations, node_resources,
                                             max([next_task.a, earliest, last_start]))

            last_pp_interval, last_pp_tasks = preemption_point_intervals[-1]
            last_pp_start, last_pp_end = last_pp_interval
            if last_pp_end > next_task.d:
                return None, False

            # Introduce a new instance into the taskset if necessary
            first_interval, _ = preemption_point_intervals[0]
            start_time = first_interval[0]
            last_task_start[original_taskname] = start_time
            instance = int(instance)

            if instance < instance_count[original_taskname]:
                periodic_task = taskset_lookup[original_taskname]
                task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                taskset = list(sorted(taskset + [task_instance], key=lambda task: (task.d, task.a, task.name)))

            # Add schedule information to resource schedules
            resource_intervals = defaultdict(list)
            for segment_interval, segment_task in preemption_point_intervals:
                interval = Interval(begin=segment_interval[0], end=segment_interval[1], data=segment_task)
                segment_intervals = defaultdict(IntervalTree)
                for resource in segment_task.resources:
                    segment_intervals[resource].add(interval)

                for resource in segment_task.resources:
                    resource_intervals[resource] |= segment_intervals[resource]

                # Add the schedule information to the overall schedule
                segment_start, segment_end = segment_interval
                schedule.append((segment_start, segment_end, segment_task))

            # Update windowed resource schedules
            if taskset:
                min_chop = max(min(list(last_task_start.values())), min(list([t.a for t in taskset])))
                earliest = min_chop
                for resource in next_task.resources:
                    resource_interval_tree = IntervalTree(Interval(begin, end, data) for begin, end, data in resource_intervals[resource])
                    if any([global_resource_occupations[resource].overlap(begin, end) for begin, end, _ in
                            resource_intervals[resource]]):
                        import pdb
                        pdb.set_trace()
                    global_resource_occupations[resource] |= resource_interval_tree
                    global_resource_occupations[resource].chop(0, min_chop)
                    global_resource_occupations[resource].merge_overlaps(strict=False, data_reducer=lambda curr_task, new_task: new_task)

        # Check validity
        valid = verify_budget_schedule(original_taskset, schedule)

        taskset = original_taskset
        return schedule, valid

    def get_segment_tasks(self, task):
        segment_tasks = []
        segment_info = find_dag_task_preemption_points(task)
        comp_time = 0
        for i, segment in enumerate(segment_info):
            segment_times, segment_locked_resources, segment_resources, segment_subtasks = segment
            segment_start_offset, segment_end_offset = segment_times
            segment_duration = segment_end_offset - segment_start_offset
            segment_resources = task.resources
            segment_locked_resources = task.locked_resources
            if i == len(segment_info) - 1:
                segment_locked_resources = []
            segment_task = ResourceTask(name="{}|{}".format(task.name, i), c=segment_duration,
                                        a=task.a + comp_time, d=task.d - task.c + comp_time + segment_duration,
                                        resources=segment_resources,
                                        locked_resources=segment_locked_resources)
            segment_tasks.append(segment_task)
            comp_time += segment_duration

        return segment_tasks

    def get_start_time(self, task, resource_occupations, node_resources, earliest):
        segment_earliest = earliest
        # Find the earliest start

        while True:
            offset = segment_earliest - task.a
            task = self.remap_task_resources(task, offset, resource_occupations, node_resources)
            segment_tasks = self.get_segment_tasks(task)

            segment_intervals = []
            prev_segment = None
            attempt_earliest = segment_earliest
            comp_time = 0
            for segment_task in segment_tasks:
                # Find the earliest start
                segment_start = self.find_earliest_start(segment_task, resource_occupations, attempt_earliest)
                if not segment_intervals and segment_start != segment_earliest:
                    segment_earliest = segment_start
                    break

                attempt_earliest = segment_start + segment_task.c
                comp_time += segment_task.c
                segment_intervals.append((segment_start, segment_start + segment_task.c))
                new_earliest = segment_earliest

                if segment_start + segment_task.c - segment_intervals[0][0] > comp_time + task.k:
                    segment_earliest = segment_start - comp_time + segment_task.c - task.k
                    break

                if prev_segment:
                    for locked_resource in prev_segment.locked_resources:
                        if resource_occupations[locked_resource][segment_intervals[-1][1]:segment_start]:
                            new_earliest = max(new_earliest, segment_start)

                if new_earliest != segment_earliest:
                    break

                prev_segment = segment_task

                if len(segment_intervals) == len(segment_tasks):
                    start_times = list(zip(segment_intervals, [segment_task for segment_task in segment_tasks]))
                    return start_times

    def find_earliest_start(self, task, resource_occupations, earliest):
        start = earliest
        distance_to_free = 1
        while distance_to_free:
            distance_to_free = 0
            sched_interval = Interval(start, start + task.c)
            for resource in task.resources:
                locking_intervals = sorted(resource_occupations[resource][0:sched_interval.begin])
                if locking_intervals:
                    last_task = locking_intervals[-1].data
                    if resource in last_task.locked_resources:
                        unlocking_intervals = sorted(resource_occupations[resource][sched_interval.begin:float('inf')])
                        unlocking_interval = unlocking_intervals[0]
                        distance_to_free = max(distance_to_free, sched_interval.begin, unlocking_interval.end)

                intervals = sorted(resource_occupations[resource][sched_interval.begin:sched_interval.end])
                if intervals:
                    overlapping_interval = intervals[0]
                    distance_to_free = max(distance_to_free, overlapping_interval.end - start)

            start += distance_to_free

        return start


class MultiResourceConsiderateFixedPointSegmentBlockPreemptionBudgetScheduler(MultiResourceConsiderateFixedPointBlockPreemptionBudgetScheduler):
    def schedule_tasks(self, taskset, topology):
        original_taskset = taskset
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        logger.debug("Computed hyperperiod {}".format(hyperperiod))
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        last_task_start = dict([(t.name, 0) for t in original_taskset])
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])

        taskset = self.initialize_taskset(taskset)

        global_resource_occupations = defaultdict(IntervalTree)
        node_resources = topology[1].nodes

        # First sort the taskset by activation time
        logger.debug("Sorting tasks by earliest deadlines")
        taskset = list(sorted(taskset, key=lambda task: (task.d, task.a, task.name)))

        schedule = []
        earliest = 0
        while taskset:
            next_task = taskset.pop(0)
            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            preemption_point_intervals = self.get_start_time(next_task, global_resource_occupations, node_resources,
                                             max([next_task.a, earliest, last_start]))

            last_pp_interval, last_pp_tasks = preemption_point_intervals[-1]
            last_pp_start, last_pp_end = last_pp_interval
            if last_pp_end > next_task.d:
                return None, False

            # Introduce a new instance into the taskset if necessary
            first_interval, _ = preemption_point_intervals[0]
            start_time = first_interval[0]
            last_task_start[original_taskname] = start_time
            instance = int(instance)

            if instance < instance_count[original_taskname]:
                periodic_task = taskset_lookup[original_taskname]
                task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                taskset = list(sorted(taskset + [task_instance], key=lambda task: (task.d, task.a, task.name)))

            # Add schedule information to resource schedules
            resource_intervals = defaultdict(list)
            for segment_interval, segment_task in preemption_point_intervals:
                interval = Interval(begin=segment_interval[0], end=segment_interval[1], data=segment_task)
                segment_intervals = defaultdict(IntervalTree)
                for resource in segment_task.resources:
                    segment_intervals[resource].add(interval)

                for resource in segment_task.resources:
                    resource_intervals[resource] |= segment_intervals[resource]

                # Add the schedule information to the overall schedule
                segment_start, segment_end = segment_interval
                schedule.append((segment_start, segment_end, segment_task))

            # Update windowed resource schedules
            if taskset:
                min_chop = max(min(list(last_task_start.values())), min(list([t.a for t in taskset])))
                earliest = min_chop
                for resource in next_task.resources:
                    resource_interval_tree = IntervalTree(Interval(begin, end, data) for begin, end, data in resource_intervals[resource])
                    if any([global_resource_occupations[resource].overlap(begin, end) for begin, end, _ in
                            resource_intervals[resource]]):
                        import pdb
                        pdb.set_trace()
                    global_resource_occupations[resource] |= resource_interval_tree
                    global_resource_occupations[resource].chop(0, min_chop)
                    global_resource_occupations[resource].merge_overlaps(strict=False, data_reducer=lambda curr_task, new_task: new_task)

        # Check validity
        valid = verify_segmented_budget_schedule(original_taskset, schedule)

        taskset = original_taskset
        return schedule, valid

    def get_start_time(self, task, resource_occupations, node_resources, earliest):
        segment_earliest = earliest
        # Find the earliest start

        while True:
            offset = segment_earliest - task.a
            task = self.remap_task_resources(task, offset, resource_occupations, node_resources)
            segment_tasks = self.get_segment_tasks(task)

            segment_intervals = []
            prev_segment = None
            attempt_earliest = segment_earliest
            comp_time = 0
            for segment_task in segment_tasks:
                # Find the earliest start
                segment_start = self.find_earliest_start(segment_task, resource_occupations, attempt_earliest)
                if not segment_intervals and segment_start != segment_earliest:
                    segment_earliest = segment_start
                    break

                attempt_earliest = segment_start + segment_task.c
                comp_time += segment_task.c
                segment_intervals.append((segment_start, segment_start + segment_task.c))
                new_earliest = segment_earliest
                if segment_start + segment_task.c - segment_intervals[0][0] > comp_time + task.k:
                    segment_earliest = segment_start - comp_time + segment_task.c - task.k
                    break

                if prev_segment:
                    for locked_resource in prev_segment.locked_resources:
                        if resource_occupations[locked_resource][segment_intervals[-1][1]:segment_start]:
                            new_earliest = max(new_earliest, segment_start)

                if new_earliest != segment_earliest:
                    break

                prev_segment = segment_task

                if len(segment_intervals) == len(segment_tasks):
                    start_times = list(zip(segment_intervals, [segment_task for segment_task in segment_tasks]))
                    return start_times

    def get_segment_tasks(self, task):
        segment_tasks = []
        segment_info = find_dag_task_preemption_points(task)
        comp_time = 0
        for i, segment in enumerate(segment_info):
            segment_times, segment_locked_resources, segment_resources, segment_subtasks = segment
            segment_start_offset, segment_end_offset = segment_times
            segment_duration = segment_end_offset - segment_start_offset
            # TODO: Make sure any tasks that produce the final pair of qubits do not lock the resource, in this case
            # they may not be in the last segment
            if i == len(segment_info) - 1:
                segment_locked_resources = []
            segment_task = ResourceTask(name="{}|{}".format(task.name, i), c=segment_duration,
                                        a=task.a + comp_time, d=task.d - task.c + comp_time + segment_duration,
                                        resources=segment_resources,
                                        locked_resources=segment_locked_resources)
            segment_tasks.append(segment_task)
            comp_time += segment_duration
        return segment_tasks

    def find_earliest_start(self, task, resource_occupations, earliest):
        start = earliest
        distance_to_free = 1
        while distance_to_free:
            distance_to_free = 0
            sched_interval = Interval(start, start + task.c)
            for resource in task.resources:
                locking_intervals = sorted(resource_occupations[resource][0:sched_interval.begin])
                if locking_intervals:
                    last_task = locking_intervals[-1].data
                    if resource in last_task.locked_resources:
                        unlocking_intervals = sorted(resource_occupations[resource][sched_interval.begin:float('inf')])
                        if not unlocking_intervals:
                            import pdb
                            pdb.set_trace()
                        unlocking_interval = unlocking_intervals[0]
                        distance_to_free = max(distance_to_free, unlocking_interval.end - sched_interval.begin)

                intervals = sorted(resource_occupations[resource][sched_interval.begin:sched_interval.end])
                if intervals:
                    overlapping_interval = intervals[0]
                    distance_to_free = max(distance_to_free, overlapping_interval.end - start)

            start += distance_to_free

        return start


class MultiResourceConsiderateFixedPointSegmentPreemptionBudgetScheduler(MultiResourceConsiderateFixedPointSegmentBlockPreemptionBudgetScheduler):
    def schedule_tasks(self, taskset, topology):
        original_taskset = taskset
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        logger.debug("Computed hyperperiod {}".format(hyperperiod))
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        last_task_start = dict([(t.name, 0) for t in original_taskset])
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])

        taskset = self.initialize_taskset(taskset)

        global_resource_occupations = defaultdict(IntervalTree)
        node_resources = topology[1].nodes

        # First sort the taskset by activation time
        logger.debug("Sorting tasks by earliest deadlines")
        taskset = list(sorted(taskset, key=lambda task: (task.d, task.a, task.name)))

        schedule = []
        earliest = 0
        while taskset:
            next_task = taskset.pop(0)
            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            preemption_point_intervals = self.get_start_time(next_task, global_resource_occupations, node_resources,
                                             max([next_task.a, earliest, last_start]))

            last_pp_interval, last_pp_tasks = preemption_point_intervals[-1]
            last_pp_start, last_pp_end = last_pp_interval
            if last_pp_end > next_task.d:
                return None, False

            # Introduce a new instance into the taskset if necessary
            first_interval, _ = preemption_point_intervals[0]
            start_time = first_interval[0]
            last_task_start[original_taskname] = start_time
            instance = int(instance)

            if instance < instance_count[original_taskname]:
                periodic_task = taskset_lookup[original_taskname]
                task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                taskset = list(sorted(taskset + [task_instance], key=lambda task: (task.d, task.a, task.name)))

            # Add schedule information to resource schedules
            resource_intervals = defaultdict(list)
            for segment_interval, segment_task in preemption_point_intervals:
                segment_start, segment_end = segment_interval
                segment_interval_list = [(resource, itree) for resource, itree in
                                         segment_task.get_resource_intervals().items()]
                segment_intervals = list(sorted(segment_interval_list, key=lambda ri: ri[1].begin()))
                for resource, itree in segment_intervals:
                    offset_itree = IntervalTree(
                        [Interval(i.begin + segment_start - segment_task.a, i.end + segment_start - segment_task.a, segment_task) for
                         i in itree])
                    resource_intervals[resource] |= offset_itree

                # Add the schedule information to the overall schedule
                segment_start, segment_end = segment_interval
                schedule.append((segment_start, segment_end, segment_task))

            # Update windowed resource schedules
            if taskset:
                min_chop = max(min(list(last_task_start.values())), min(list([t.a for t in taskset])))
                earliest = min_chop
                for resource in next_task.resources:
                    resource_interval_tree = IntervalTree(Interval(begin, end, data) for begin, end, data in resource_intervals[resource])
                    last_task = sorted(resource_interval_tree)[-1].data
                    if resource in last_task.locked_resources:
                        last_task.locked_resources.remove(resource)

                    if any([global_resource_occupations[resource].overlap(begin, end) for begin, end, _ in
                            resource_intervals[resource]]):
                        import pdb
                        pdb.set_trace()
                    global_resource_occupations[resource] |= resource_interval_tree
                    global_resource_occupations[resource].chop(0, min_chop)
                    global_resource_occupations[resource].merge_overlaps(strict=False, data_reducer=lambda curr_task, new_task: new_task)

        # Check validity
        valid = verify_segmented_budget_schedule(original_taskset, schedule)

        taskset = original_taskset
        return schedule, valid

    def get_start_time(self, task, resource_occupations, node_resources, earliest):
        segment_earliest = earliest
        # Find the earliest start

        while True:
            offset = segment_earliest - task.a
            task = self.remap_task_resources(task, offset, resource_occupations, node_resources)
            segment_tasks = self.get_segment_tasks(task)

            segment_intervals = []
            prev_segment = None
            attempt_earliest = segment_earliest
            comp_time = 0
            for segment_task in segment_tasks:
                # Find the earliest start
                segment_start = self.find_earliest_start(segment_task, resource_occupations, attempt_earliest)
                if not segment_intervals and segment_start != segment_earliest:
                    segment_earliest = segment_start
                    break

                attempt_earliest = segment_start + segment_task.c
                comp_time += segment_task.c
                segment_intervals.append((segment_start, segment_start + segment_task.c))
                new_earliest = segment_earliest
                if segment_start + segment_task.c - segment_intervals[0][0] > comp_time + task.k:
                    segment_earliest = segment_start - comp_time + segment_task.c - task.k
                    break

                if prev_segment:
                    for locked_resource in prev_segment.locked_resources:
                        if resource_occupations[locked_resource][segment_intervals[-1][1]:segment_start]:
                            new_earliest = max(new_earliest, segment_start)

                if new_earliest != segment_earliest:
                    break

                prev_segment = segment_task

                if len(segment_intervals) == len(segment_tasks):
                    start_times = list(zip(segment_intervals, [segment_task for segment_task in segment_tasks]))
                    return start_times

    def get_segment_tasks(self, task):
        segment_tasks = []
        segment_info = find_dag_task_preemption_points(task)
        comp_time = 0
        for i, segment in enumerate(segment_info):
            segment_times, segment_locked_resources, segment_resources, segment_subtasks = segment
            segment_start_offset, segment_end_offset = segment_times
            segment_duration = segment_end_offset - segment_start_offset
            segment_task = ResourceDAGTask(name="{}|{}".format(task.name, i), a=segment_start_offset,
                                           d=max([segment_subtask.d for segment_subtask in segment_subtasks]),
                                           tasks=segment_subtasks)
            if i == len(segment_info) - 1:
                segment_task.locked_resources = []
            else:
                segment_task.locked_resources = segment_locked_resources
            segment_tasks.append(segment_task)
            comp_time += segment_duration

        return segment_tasks

    def find_earliest_start(self, task, resource_occupations, earliest):
        start = earliest
        distance_to_free = 1
        while distance_to_free:
            distance_to_free = 0
            resource_interval_list = [(resource, itree) for resource, itree in task.get_resource_intervals().items()]
            resource_intervals = list(sorted(resource_interval_list, key=lambda ri: ri[1].begin()))
            offset = start - task.a
            for resource, itree in resource_intervals:
                for interval in itree:
                    locking_intervals = sorted(resource_occupations[resource][0:interval.begin + offset])
                    if locking_intervals:
                        last_task = locking_intervals[-1].data
                        if resource in last_task.locked_resources:
                            unlocking_intervals = sorted(resource_occupations[resource][interval.begin + offset:float('inf')])
                            if not unlocking_intervals:
                                import pdb
                                pdb.set_trace()
                            unlocking_interval = unlocking_intervals[0]
                            distance_to_free = max(distance_to_free, unlocking_interval.end - (interval.begin+offset))

                    intervals = sorted(resource_occupations[resource][interval.begin+offset:interval.end+offset])
                    if intervals:
                        overlapping_interval = intervals[0]
                        distance_to_free = max(distance_to_free, overlapping_interval.end - start)

            start += distance_to_free

        return start


class MultipleResourceInconsiderateBlockPreemptionBudgetScheduler(Scheduler):
    internal_scheduler_class = MultiResourceInconsiderateFixedPointBlockPreemptionBudgetScheduler
    def schedule_tasks(self, dagset, topology):
        # Convert DAGs into tasks
        tasks = {}
        resources = set()
        for dag_task in dagset:
            block_task = PeriodicBudgetResourceDAGTask(name=dag_task.name, tasks=dag_task.subtasks, p=dag_task.p,
                                                       k=int(dag_task.k))
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
        scheduler = self.internal_scheduler_class()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule_tasks(taskset, topology)
            schedules.append((taskset, schedule, valid))

        # Set of schedules is the schedule for each group of resources
        return schedules


class MultipleResourceInconsiderateSegmentBlockPreemptionBudgetScheduler(MultipleResourceInconsiderateBlockPreemptionBudgetScheduler):
    internal_scheduler_class = MultiResourceInconsiderateFixedPointSegmentBlockPreemptionBudgetScheduler


class MultipleResourceInconsiderateSegmentPreemptionBudgetScheduler(MultipleResourceInconsiderateBlockPreemptionBudgetScheduler):
    internal_scheduler_class = MultiResourceInconsiderateFixedPointSegmentPreemptionBudgetScheduler


class MultipleResourceConsiderateBlockPreemptionBudgetScheduler(MultipleResourceInconsiderateSegmentBlockPreemptionBudgetScheduler):
    internal_scheduler_class = MultiResourceConsiderateFixedPointBlockPreemptionBudgetScheduler


class MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler(MultipleResourceInconsiderateSegmentBlockPreemptionBudgetScheduler):
    internal_scheduler_class = MultiResourceConsiderateFixedPointSegmentBlockPreemptionBudgetScheduler

class MultipleResourceConsiderateSegmentPreemptionBudgetScheduler(MultipleResourceInconsiderateSegmentBlockPreemptionBudgetScheduler):
    internal_scheduler_class = MultiResourceConsiderateFixedPointSegmentPreemptionBudgetScheduler