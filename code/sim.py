import pdb
from task import Task, PeriodicTask, DAGResourceSubTask, PeriodicResourceDAGTask, ResourceTask, generate_non_periodic_task_set, generate_non_periodic_budget_task_set, ResourceDAGTask, PeriodicBudgetTask
from scheduler import NPEDFScheduler, PeriodicNPEDFScheduler, OptimalScheduler, PeriodicOptimalScheduler, MultipleResourceBlockNPEDFScheduler, \
    MultipleResourceOptimalBlockScheduler, MultipleResourceBlockCEDFScheduler, PeriodicCEDFScheduler, pretty_print_schedule, compute_schedule_lateness, \
    check_wc_np_feasibility, MultiResourceNPEDFScheduler, PeriodicEDFScheduler, PeriodicPreemptionBudgetScheduler, PeriodicEDFScheduler, PeriodicPreemptionBudgetSchedulerNew
from time import time

def optimal_schedule():
    tasks = [
        Task(name="T1", c=1, d=4),
        Task(name="T2", c=2, d=3),
        Task(name="T3", c=4, d=7),
        Task(name="T4", c=3, d=9)
    ]

    scheduler = OptimalScheduler()
    schedule, valid = scheduler.schedule(tasks)
    pdb.set_trace()


def task_schedule():
    tasks = [
        Task(name="T1", a=0, c=2, d=4),
        Task(name="T2", a=1, c=2, d=3)
    ]

    scheduler = NPEDFScheduler()
    schedule, valid = scheduler.schedule(tasks)
    pdb.set_trace()


def periodic_optimal_schedule():
    periodic_tasks = [
        PeriodicTask(name="T1", c=1, p=5),
        PeriodicTask(name="T2", c=1, p=6),
        PeriodicTask(name="T3", c=1, p=7),
        PeriodicTask(name="T4", c=1, p=8)
    ]

    scheduler = PeriodicOptimalScheduler()
    schedule, valid = scheduler.schedule(periodic_tasks)
    pdb.set_trace()


def periodic_task_schedule():
    periodic_tasks = [
        PeriodicTask(name="T1", c=1, p=5),
        PeriodicTask(name="T2", c=1, p=6),
        PeriodicTask(name="T3", c=1, p=7),
        PeriodicTask(name="T4", c=1, p=8)
    ]

    scheduler = PeriodicNPEDFScheduler()
    schedule, valid = scheduler.schedule(periodic_tasks)
    pdb.set_trace()


def multiple_resource_non_preemptive_block_scheduling():
    resources = ["R{}".format(i) for i in range(10)]

    # Task 1
    # Subtask 1 with resources R0, R1
    t1st1 = DAGResourceSubTask(name="T1,1", c=1, resources=resources[0:2])

    # Subtask 2 with resources R1, R2
    t1st2 = DAGResourceSubTask(name="T1,2", c=2, resources=resources[1:3])

    # Subtask 3 with resources R0, R1
    t1st3 = DAGResourceSubTask(name="T1,3", c=1, resources=resources[0:2])

    t1st1.set_children([t1st2])
    t1st2.set_parents([t1st1])
    t1st2.set_children([t1st3])
    t1st3.set_parents([t1st2])
    t1 = PeriodicResourceDAGTask(name="T1", tasks=[t1st1, t1st2, t1st3], p=16)

    # Task 2
    # Subtask 1 with resources R2, R3, R4
    t2st1 = DAGResourceSubTask(name="T2,1", c=2, resources=resources[2:5])

    # Subtask 2 with resources R1, R2
    t2st2 = DAGResourceSubTask(name="T2,2", c=3, resources=resources[1:3])

    # Subtask 3 with resources R0, R1, R4, R5
    t2st3 = DAGResourceSubTask(name="T2,3", c=1, resources=resources[0:2] + resources[4:6])

    t2st1.set_children([t2st2])
    t2st2.set_parents([t2st1])
    t2st2.set_children([t2st3])
    t2st3.set_parents([t2st2])
    t2 = PeriodicResourceDAGTask(name="T2", tasks=[t2st1, t2st2, t2st3], p=12)

    # Task 3
    # Subtask 1 with resources R6, R7, R8
    t3st1 = DAGResourceSubTask(name="T3,1", c=4, resources=resources[6:9])

    # Subtask 2 with resources R5,R6
    t3st2 = DAGResourceSubTask(name="T3,2", c=3, resources=resources[5:7])

    # Subtask 3 with resources R8,R9
    t3st3 = DAGResourceSubTask(name="T3,3", c=2, resources=resources[8:10])

    t3st1.set_children([t3st2])
    t3st2.set_parents([t3st1])
    t3st2.set_children([t3st3])
    t3st3.set_parents([t3st2])
    t3 = PeriodicResourceDAGTask(name="T3", tasks=[t3st1, t3st2, t3st3], p=36)

    tasks = [t1, t2, t3]

    optscheduler = MultipleResourceOptimalBlockScheduler()
    optschedules = optscheduler.schedule(tasks)
    npedfscheduler = MultipleResourceBlockNPEDFScheduler()
    npedfschedules = npedfscheduler.schedule(tasks)
    cedfscheduler = MultipleResourceBlockCEDFScheduler()
    cedfschedules = cedfscheduler.schedule(tasks)
    pdb.set_trace()


def multiple_resource_non_preemptive_scheduling():
    resources = ["R{}".format(i) for i in range(10)]

    # Task 1, a=0, c=8, d=9
    # Subtask 1 with resources R0, R1
    t1st1 = DAGResourceSubTask(name="T1,1", a=0, c=10, d=15, resources=resources[0:2])

    # Extra subtask needing resource R2
    t1stextra = DAGResourceSubTask(name="T1,e", a=5, c=7, d=15, resources=[resources[2]])

    # Subtask 2 with resources R1, R2
    t1st2 = DAGResourceSubTask(name="T1,2", a=12, c=5, d=20, resources=resources[1:3])

    # Subtask 2 with resources R2, R3
    t1st3 = DAGResourceSubTask(name="T1,3", a=17, c=5, d=25, resources=resources[2:4])

    # Subtask 2 with resources R3, R4
    t1st4 = DAGResourceSubTask(name="T1,3", a=22, c=5, d=30, resources=resources[3:5])

    # Subtask 5 with resources R2, R3
    t1st5 = DAGResourceSubTask(name="T1,3", a=27, c=5, d=35, resources=resources[2:4])

    # Subtask 6 with resources R1, R2
    t1st6 = DAGResourceSubTask(name="T1,6", a=32, c=5, d=40, resources=resources[1:3])

    # Subtask 7 with resources R0, R1
    t1st7 = DAGResourceSubTask(name="T1,7", a=37, c=5, d=45, resources=resources[0:2])

    subtasks = [t1st2, t1st3, t1st4, t1st5, t1st6, t1st7]
    for s1, s2 in zip(subtasks, subtasks[1:]):
        s1.set_children([s2])
        s2.set_parents([s1])
    t1st1.set_children([t1st2])
    t1stextra.set_children([t1st2])
    t1st2.set_parents([t1st1, t1stextra])
    subtasks = [t1st1, t1stextra] + subtasks
    t1 = ResourceDAGTask(name="T1", d=45, tasks=subtasks)
    pdb.set_trace()

    # Task 2 a=0, c=3, d=6
    # Subtask 1 with resources R0, R1
    t2st1 = DAGResourceSubTask(name="T2,1", a=0, c=5, d=25, resources=resources[0:2])

    # Subtask 2 with resources R1, R2
    t2st2 = DAGResourceSubTask(name="T2,2", a=5, c=5, d=30, resources=resources[1:3])

    # Subtask 3 with resources R0, R1
    t2st3 = DAGResourceSubTask(name="T2,3", a=10, c=5, d=35, resources=resources[0:2])

    t2st1.set_children([t2st2])
    t2st2.set_parents([t2st1])
    t2st2.set_children([t2st3])
    t2st3.set_parents([t2st2])
    t2 = ResourceDAGTask(name="T2", d=30, tasks=[t2st1, t2st2, t2st3])

    tasks = [t1, t2]
    mrnpedfscheduler = MultiResourceNPEDFScheduler()
    schedules, valid = mrnpedfscheduler.schedule(tasks)
    pdb.set_trace()


def cedf_schedule_test():
    periodic_tasks = [
        PeriodicTask(name="T1", a=0, c=15, p=60),
        PeriodicTask(name="T2", a=0, c=60, p=80),
    ]

    print(check_wc_np_feasibility(periodic_tasks))
    pdb.set_trace()
    print("Computing Feasible")
    optscheduler = PeriodicOptimalScheduler()
    start = time()
    optschedule, valid = optscheduler.schedule(periodic_tasks)
    end = time()
    print("Took {}s".format(end-start))
    if not valid:
        print("Infeasible!")
    else:
        pretty_print_schedule(optschedule[:15])
        print("Lateness: {}".format(compute_schedule_lateness(optschedule)))

    print("Computing with NPEDF")
    npedfscheduler = PeriodicNPEDFScheduler()
    start = time()
    npedfschedule, valid = npedfscheduler.schedule(periodic_tasks)
    end = time()
    print("Took {}s".format(end - start))
    if not valid:
        print("Infeasible!")
    else:
        pretty_print_schedule(npedfschedule[:15])
        print("Lateness: {}".format(compute_schedule_lateness(npedfschedule)))

    print("Computing with CEDF")
    cedfscheduler = PeriodicCEDFScheduler()
    start = time()
    cedfschedule, valid = cedfscheduler.schedule(periodic_tasks)
    end = time()
    print("Took {}s".format(end - start))
    if not valid:
        print("Infeasible!")
    else:
        pretty_print_schedule(cedfschedule[:15])
        print("Lateness: {}".format(compute_schedule_lateness(cedfschedule)))

    pdb.set_trace()


def edf_schedule_test():
    tasks = [
        PeriodicTask(name="T1", c=1, p=10),
        PeriodicTask(name="T2", c=8, p=30),
        PeriodicTask(name="T3", c=17, p=60)
    ]

    edfscheduler = PeriodicEDFScheduler()
    edfschedule, edfvalid = edfscheduler.schedule(tasks)
    npedfscheduler = PeriodicNPEDFScheduler()
    npedfschedule, npedfvalid = npedfscheduler.schedule(tasks)
    pdb.set_trace()


# EDF-LBF can successfully schedule these tasks while EDF cannot
def get_pbtasks():
    return [
        PeriodicBudgetTask(name="T1", a=0, c=2, p=4, k=0),
        PeriodicBudgetTask(name="T2", a=0, c=6, p=12, k=2)
    ]


# EDF can successfully do this while EDF-LBF cannot
def get_pbtasks():
    return [
        PeriodicBudgetTask(name="T1", a=0, c=3, p=4, k=1),
        PeriodicBudgetTask(name="T2", a=0, c=3, p=12, k=4)
    ]


def preempt_budget_test():
    edf_scheduler = PeriodicEDFScheduler()
    start = time()
    edf_schedule, valid = edf_scheduler.schedule_tasks(get_pbtasks())
    print("EDF took {}".format(time() - start))

    pbedf_scheduler = PeriodicPreemptionBudgetScheduler()
    start = time()
    pbedf_schedule, pbedf_valid = pbedf_scheduler.schedule_tasks(get_pbtasks())
    print("PBEDF took {}".format(time() - start))

    pbedf_scheduler_new = PeriodicPreemptionBudgetSchedulerNew()
    start = time()
    pbedf_schedule_new, pbedf_valid_new = pbedf_scheduler_new.schedule_tasks(get_pbtasks())
    print("PBEDF_NEW took {}".format(time() - start))

    npedf_scheduler = PeriodicNPEDFScheduler()
    start = time()
    npedf_schedule, npedf_valid = npedf_scheduler.schedule_tasks(get_pbtasks())
    print("NPEDF took {}".format(time()-start))

    if not pbedf_valid:
        print("PBEDF scheduler does not satisfy budgets")

    if not pbedf_valid_new:
        print("PBEDF_NEW scheduler does not satisfy budgets")

    if not pbedf_scheduler_new.check_feasible(edf_schedule, generate_non_periodic_budget_task_set(get_pbtasks())):
        print("EDF schedule does not satisfy budgets")

    if not pbedf_scheduler_new.check_feasible(npedf_schedule, generate_non_periodic_budget_task_set(get_pbtasks())):
        print("NPEDF schedule does not satisfy deadlines")

    wcrts = pbedf_scheduler_new.get_worst_case_response_times(pbedf_schedule_new, generate_non_periodic_budget_task_set(get_pbtasks()))
    pdb.set_trace()


def main():
    preempt_budget_test()


if __name__ == "__main__":
    main()
