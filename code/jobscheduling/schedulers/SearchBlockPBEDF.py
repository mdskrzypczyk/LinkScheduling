from collections import defaultdict
from intervaltree import Interval, IntervalTree
from jobscheduling.log import LSLogger
from jobscheduling.schedulers.scheduler import CommonScheduler, BaseMultipleResourceScheduler, \
    verify_budget_schedule, verify_segmented_budget_schedule
from jobscheduling.task import find_dag_task_preemption_points, ResourceTask, ResourceDAGTask


logger = LSLogger()


class MRIFixedPointBlockPreemptionBudgetScheduler(CommonScheduler):
    def verify_schedule(self, taskset, schedule):
        """
        Verifies the schedule using a specific verification function
        :param taskset: type list
            List of PeriodicTasks to verify in the schedule
        :param schedule: type list
            List of (start, end, task) information composing the schedule
        :return: bool
            True/False
        """
        return verify_budget_schedule(taskset, schedule)

    def extract_resource_intervals_from_preemption_point_intervals(self, preemption_point_intervals):
        """
        Obtains the intervals of task execution from the preemption points of the task
        :param preemption_point_intervals: type list
            List of tuples of ((start, end), task) representing the intervals of time where the task is scheduled/
        :return: type dict(IntervalTree)
            Dictionary of interval trees representing periods of resource occupation by tasks
        """
        # Add schedule information to resource schedules
        resource_intervals = defaultdict(IntervalTree)
        for segment_interval, segment_task in preemption_point_intervals:
            interval = Interval(begin=segment_interval[0], end=segment_interval[1])
            for resource in segment_task.resources:
                resource_intervals[resource].add(interval)

        return resource_intervals

    def schedule_preemption_point_intervals(self, schedule, preemption_point_intervals):
        """
        Adds the preemption point intervals to the running schedule
        :param schedule: type list
            List of (start, end, task) information encoding the schedule
        :param preemption_point_intervals: type list
            List of ((start, end), segment task) to add to the schedule
        :return: None
        """
        for segment_interval, segment_task in preemption_point_intervals:
            segment_start, segment_end = segment_interval
            schedule.append((segment_start, segment_end, segment_task))

    def update_resource_occupations(self, resource_occupations, resource_intervals):
        """
        Updates the recorded occupation of resources in the system
        :param resource_occupations: type dict(IntervalTree)
            A dictionary from resource identifiers to interval trees representing the periods of occupation of resources
        :param resource_intervals: type dict(IntervalTree)
            A dictionary of the newly occupied time intervals to add to the recorded occupation
        :return: None
        """
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

    def construct_segment_task(self, task, segment, comp_time, i):
        """
        Constructs a Task representing a segment of the DAGTask
        :param task: type DAGTask
            The DAGTask to construct the segment for
        :param segment: type list
            List of information describing the segment, (start, end), resources locked by segment, resources needed
            by segment, DAGSubTasks of DAGTask that belong to the segment
        :param comp_time: type int
            The amount of computation time performed up to this segment
        :param i: type int
            An enumeration of the segment
        :return: type Task
            A Task representing the set of subtasks belonging to the segment
        """
        segment_times, segment_locked_resources, segment_resources, segment_subtasks = segment
        segment_start_offset, segment_end_offset = segment_times
        segment_duration = segment_end_offset - segment_start_offset

        segment_task = ResourceTask(name="{}|{}".format(task.name, i), c=segment_duration,
                                    a=task.a + comp_time, d=task.d - task.c + comp_time + segment_duration,
                                    resources=task.resources, locked_resources=task.resources)

        return segment_task

    def find_earliest_start(self, task, resource_occupations, earliest):
        """
        Finds the start time for a task
        :param task: type DAGTask
            The task to obtain the start time for
        :param resource_occupations: type dict(IntervalTree)
            A dictionary of resource identifiers to interval trees representing resource occupations
        :param earliest: type int
            The earliest point in time the task is permitted to start
        :return: type int
            The earliest time the incoming task may start
        """
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


class MRIFixedPointSegmentBlockPreemptionBudgetScheduler(MRIFixedPointBlockPreemptionBudgetScheduler):
    def verify_schedule(self, taskset, schedule):
        """
        Verifies the schedule using a specific verification function
        :param taskset: type list
            List of PeriodicTasks to verify in the schedule
        :param schedule: type list
            List of (start, end, task) information composing the schedule
        :return: bool
            True/False
        """
        return verify_segmented_budget_schedule(taskset, schedule)

    def construct_segment_task(self, task, segment, comp_time, i):
        """
        Constructs a Task representing a segment of the DAGTask
        :param task: type DAGTask
            The DAGTask to construct the segment for
        :param segment: type list
            List of information describing the segment, (start, end), resources locked by segment, resources needed
            by segment, DAGSubTasks of DAGTask that belong to the segment
        :param comp_time: type int
            The amount of computation time performed up to this segment
        :param i: type int
            An enumeration of the segment
        :return: type Task
            A Task representing the set of subtasks belonging to the segment
        """
        segment_times, segment_locked_resources, segment_resources, segment_subtasks = segment
        segment_start_offset, segment_end_offset = segment_times
        segment_duration = segment_end_offset - segment_start_offset
        segment_task = ResourceTask(name="{}|{}".format(task.name, i), c=segment_duration,
                                    a=task.a + comp_time, d=task.d - task.c + comp_time + segment_duration,
                                    resources=segment_resources,
                                    locked_resources=segment_locked_resources)
        return segment_task


class MRIFixedPointSegmentPreemptionBudgetScheduler(MRIFixedPointSegmentBlockPreemptionBudgetScheduler):
    def extract_resource_intervals_from_preemption_point_intervals(self, preemption_point_intervals):
        """
        Obtains the intervals of task execution from the preemption points of the task
        :param preemption_point_intervals: type list
            List of tuples of ((start, end), task) representing the intervals of time where the task is scheduled/
        :return: type dict(IntervalTree)
            Dictionary of interval trees representing periods of resource occupation by tasks
        """
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

    def construct_segment_task(self, task, segment, comp_time, i):
        """
        Constructs a Task representing a segment of the DAGTask
        :param task: type DAGTask
            The DAGTask to construct the segment for
        :param segment: type list
            List of information describing the segment, (start, end), resources locked by segment, resources needed
            by segment, DAGSubTasks of DAGTask that belong to the segment
        :param comp_time: type int
            The amount of computation time performed up to this segment
        :param i: type int
            An enumeration of the segment
        :return: type Task
            A Task representing the set of subtasks belonging to the segment
        """
        segment_times, segment_locked_resources, segment_resources, segment_subtasks = segment
        segment_start_offset, segment_end_offset = segment_times
        segment_task = ResourceDAGTask(name="{}|{}".format(task.name, i), a=segment_start_offset,
                                       d=max([segment_subtask.d for segment_subtask in segment_subtasks]),
                                       tasks=segment_subtasks)

        return segment_task


class MRCFixedPointBlockPreemptionBudgetScheduler(MRIFixedPointBlockPreemptionBudgetScheduler):
    def update_resource_occupations(self, resource_occupations, resource_intervals):
        """
        Updates the recorded occupation of resources in the system
        :param resource_occupations: type dict(IntervalTree)
            A dictionary from resource identifiers to interval trees representing the periods of occupation of resources
        :param resource_intervals: type dict(IntervalTree)
            A dictionary of the newly occupied time intervals to add to the recorded occupation
        :return: None
        """
        # Update windowed resource schedules
        for resource in resource_intervals.keys():
            resource_interval_tree = IntervalTree(
                Interval(begin, end, data) for begin, end, data in resource_intervals[resource])
            if any([resource_occupations[resource].overlap(begin, end) for begin, end, _ in
                    resource_intervals[resource]]):
                import pdb
                pdb.set_trace()
            resource_occupations[resource] |= resource_interval_tree
            resource_occupations[resource].merge_overlaps(strict=False,
                                                          data_reducer=lambda curr_task, new_task: new_task)

    def extract_resource_intervals_from_preemption_point_intervals(self, preemption_point_intervals):
        """
        Obtains the intervals of task execution from the preemption points of the task
        :param preemption_point_intervals: type list
            List of tuples of ((start, end), task) representing the intervals of time where the task is scheduled/
        :return: type dict(IntervalTree)
            Dictionary of interval trees representing periods of resource occupation by tasks
        """
        # Add schedule information to resource schedules
        resource_intervals = defaultdict(IntervalTree)
        for segment_interval, segment_task in preemption_point_intervals:
            interval = Interval(begin=segment_interval[0], end=segment_interval[1], data=segment_task)
            for resource in segment_task.resources:
                resource_intervals[resource].add(interval)

        return resource_intervals

    def get_segment_tasks(self, task):
        """
        Gets the set of tasks representing the segments of the provided task
        :param task: type DAGTask
            The task to obtain segment tasks for
        :return: type list
            List of Task objects that describe the segments of the DAGTask
        """
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
        segment_earliest = earliest
        # Find the earliest start

        while True:
            # task = self.remap_task_resources(task, offset, resource_occupations, node_resources)
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
        """
        Finds the start time for a task
        :param task: type DAGTask
            The task to obtain the start time for
        :param resource_occupations: type dict(IntervalTree)
            A dictionary of resource identifiers to interval trees representing resource occupations
        :param earliest: type int
            The earliest point in time the task is permitted to start
        :return: type int
            The earliest time the incoming task may start
        """
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


class MRCFixedPointSegmentBlockPreemptionBudgetScheduler(MRCFixedPointBlockPreemptionBudgetScheduler):
    def verify_schedule(self, taskset, schedule):
        """
        Verifies the schedule using a specific verification function
        :param taskset: type list
            List of PeriodicTasks to verify in the schedule
        :param schedule: type list
            List of (start, end, task) information composing the schedule
        :return: bool
            True/False
        """
        return verify_segmented_budget_schedule(taskset, schedule)

    def update_resource_occupations(self, resource_occupations, resource_intervals):
        """
        Updates the recorded occupation of resources in the system
        :param resource_occupations: type dict(IntervalTree)
            A dictionary from resource identifiers to interval trees representing the periods of occupation of resources
        :param resource_intervals: type dict(IntervalTree)
            A dictionary of the newly occupied time intervals to add to the recorded occupation
        :return: None
        """
        # Update windowed resource schedules
        for resource in resource_intervals.keys():
            resource_interval_tree = IntervalTree(
                Interval(begin, end, data) for begin, end, data in resource_intervals[resource])
            last_task = sorted(resource_interval_tree)[-1].data
            if resource in last_task.locked_resources:
                last_task.locked_resources.remove(resource)
            if any([resource_occupations[resource].overlap(begin, end) for begin, end, _ in
                    resource_intervals[resource]]):
                import pdb
                pdb.set_trace()
            resource_occupations[resource] |= resource_interval_tree
            resource_occupations[resource].merge_overlaps(strict=False,
                                                          data_reducer=lambda curr_task, new_task: new_task)

    def get_segment_tasks(self, task):
        """
        Gets the set of tasks representing the segments of the provided task
        :param task: type DAGTask
            The task to obtain segment tasks for
        :return: type list
            List of Task objects that describe the segments of the DAGTask
        """
        segment_tasks = []
        segment_info = find_dag_task_preemption_points(task)
        comp_time = 0
        for i, segment in enumerate(segment_info):
            segment_times, segment_locked_resources, segment_resources, segment_subtasks = segment
            segment_start_offset, segment_end_offset = segment_times
            segment_duration = segment_end_offset - segment_start_offset
            if i == len(segment_info) - 1:
                segment_locked_resources = []

            segment_task = ResourceTask(name="{}|{}".format(task.name, i), c=segment_duration,
                                        a=task.a + comp_time, d=task.d - task.c + comp_time + segment_duration,
                                        resources=segment_resources,
                                        locked_resources=segment_locked_resources)
            segment_tasks.append(segment_task)
            comp_time += segment_duration
        return segment_tasks


class MRCFixedPointSegmentPreemptionBudgetScheduler(MRCFixedPointSegmentBlockPreemptionBudgetScheduler):
    def extract_resource_intervals_from_preemption_point_intervals(self, preemption_point_intervals):
        """
        Obtains the intervals of task execution from the preemption points of the task
        :param preemption_point_intervals: type list
            List of tuples of ((start, end), task) representing the intervals of time where the task is scheduled/
        :return: type dict(IntervalTree)
            Dictionary of interval trees representing periods of resource occupation by tasks
        """
        resource_intervals = defaultdict(IntervalTree)
        for segment_interval, segment_task in preemption_point_intervals:
            segment_start, segment_end = segment_interval
            segment_interval_list = [(resource, itree) for resource, itree in
                                     segment_task.get_resource_intervals().items()]
            segment_intervals = list(sorted(segment_interval_list, key=lambda ri: ri[1].begin()))
            for resource, itree in segment_intervals:
                offset_itree = IntervalTree([Interval(i.begin + segment_start - segment_task.a,
                                                      i.end + segment_start - segment_task.a, segment_task)
                                             for i in itree])
                resource_intervals[resource] |= offset_itree

        return resource_intervals

    def get_segment_tasks(self, task):
        """
        Gets the set of tasks representing the segments of the provided task
        :param task: type DAGTask
            The task to obtain segment tasks for
        :return: type list
            List of Task objects that describe the segments of the DAGTask
        """
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
        """
        Finds the start time for a task
        :param task: type DAGTask
            The task to obtain the start time for
        :param resource_occupations: type dict(IntervalTree)
            A dictionary of resource identifiers to interval trees representing resource occupations
        :param earliest: type int
            The earliest point in time the task is permitted to start
        :return: type int
            The earliest time the incoming task may start
        """
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
                            unlock_start = interval.begin + offset
                            unlocking_intervals = sorted(resource_occupations[resource][unlock_start:float('inf')])
                            if not unlocking_intervals:
                                import pdb
                                pdb.set_trace()
                            unlocking_interval = unlocking_intervals[0]
                            distance_to_free = max(distance_to_free, unlocking_interval.end - (interval.begin + offset))

                    intervals = sorted(resource_occupations[resource][interval.begin + offset:interval.end + offset])
                    if intervals:
                        overlapping_interval = intervals[0]
                        distance_to_free = max(distance_to_free, overlapping_interval.end - start)

            start += distance_to_free

        return start


class MultipleResourceInconsiderateBlockPreemptionBudgetScheduler(BaseMultipleResourceScheduler):
    internal_scheduler_class = MRIFixedPointBlockPreemptionBudgetScheduler


class MultipleResourceInconsiderateSegmentBlockPreemptionBudgetScheduler(BaseMultipleResourceScheduler):
    internal_scheduler_class = MRIFixedPointSegmentBlockPreemptionBudgetScheduler


class MultipleResourceInconsiderateSegmentPreemptionBudgetScheduler(BaseMultipleResourceScheduler):
    internal_scheduler_class = MRIFixedPointSegmentPreemptionBudgetScheduler


class MultipleResourceConsiderateBlockPreemptionBudgetScheduler(BaseMultipleResourceScheduler):
    internal_scheduler_class = MRCFixedPointBlockPreemptionBudgetScheduler


class MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler(BaseMultipleResourceScheduler):
    internal_scheduler_class = MRCFixedPointSegmentBlockPreemptionBudgetScheduler


class MultipleResourceConsiderateSegmentPreemptionBudgetScheduler(BaseMultipleResourceScheduler):
    internal_scheduler_class = MRCFixedPointSegmentPreemptionBudgetScheduler
