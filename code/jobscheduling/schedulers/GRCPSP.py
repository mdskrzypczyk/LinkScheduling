from collections import defaultdict
from copy import copy
from jobscheduling.schedulers.scheduler import get_lcm_for
from jobscheduling.task import ResourceTask

class MultipleResourceBlockGRCPSPScheduler:
    def create_new_task_instance(self, periodic_task, instance):
        dag_copy = copy(periodic_task)
        release_offset = dag_copy.a + dag_copy.p * instance
        dag_copy.a = release_offset
        for subtask in dag_copy.subtasks:
            subtask.a += release_offset
        dag_instance = ResourceTask(name="{}|{}".format(dag_copy.name, instance), a=release_offset, c=dag_copy.c,
                                          d=release_offset + dag_copy.p, resources=dag_copy.resources)
        return dag_instance

    def schedule_tasks(self, taskset, topology):
        hyperperiod = get_lcm_for([t.p for t in taskset])
        T = hyperperiod
        p = 0
        m = mprime = 0
        FS = defaultdict(dict)
        l = {}
        Q = defaultdict(int)
        S = {1}
        PS = {1}
        C = set()
        s = {}
        f = {1: 0}
        d = {1: 0}
        jobid = 2
        for periodic_task in taskset:
            for instance in range(hyperperiod // periodic_task.p):
                task_instance = self.create_new_task_instance(periodic_task, instance)
                FS[1][jobid] = task_instance.a
                d[jobid] = task_instance.c
                C |= {jobid}
                s[jobid] = task_instance.a
                l[jobid] = task_instance.d - task_instance.c
                Q[jobid] = task_instance.c
                jobid += 1
                # No precedence relations among jobs, skip FSprime calculation

        dummy_id = jobid

        self.helper(T, p, m, mprime, FS, l, Q, S, PS, C, s, f, d, dummy_id)

    def helper(self, T, p, m, mprime, FS, l, Q, S, PS, C, s, f, d, dummy_id):
        if dummy_id in PS:
            return PS

        # Augment
        m = min(s[x] for x in C)
        S = set(filter(lambda j: f[j] <= m, S))

        # TODO: Check if current cutset is dominated

        E = {x for x in C if s[x] == m}

        # Schedule
        PS |= {i for i in E}
        S |= {i for i in E}
        for i in E:
            f[i] = m + d[i]

        # TODO: Track successors of jobs
        C = C - E





    def augment(self):
