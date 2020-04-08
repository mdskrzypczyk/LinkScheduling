import matplotlib.pyplot as plt
import time
from math import ceil
from collections import defaultdict
from jobscheduling.log import LSLogger
from jobscheduling.protocols import convert_protocol_to_task, schedule_dag_for_resources
from jobscheduling.task import get_lcm_for, get_gcd_for
from jobscheduling.visualize import draw_DAG, schedule_timeline, resource_timeline, schedule_and_resource_timelines, protocol_timeline
from jobscheduling.topology import gen_line_topology
from simulations.common import get_schedulers, get_network_demands, get_protocol_without_rate_constraint, \
    balance_taskset_resource_utilization, check_resource_utilization, select_rate


logger = LSLogger()


def preemption_budget_allowance():
    pass


def main():
    num_network_nodes = 7
    num_tasksets = 1
    line_topology = gen_line_topology(num_network_nodes, num_comm_q=1, num_storage_q=3)
    slot_size = 0.05
    demand_size = 100

    network_schedulers = get_schedulers()
    results = {}
    network_tasksets = []
    network_taskset_properties = []

    for i in range(num_tasksets):
        logger.info("Generating taskset {}".format(i))
        taskset_properties = {}

        # Generate task sets according to some utilization characteristics and preemption budget allowances
        try:
            demands = get_network_demands(line_topology, demand_size)

            taskset = []
            num_succ = 0
            for demand in demands:
                try:
                    logger.debug("Constructing protocol for request {}".format(demand))
                    protocol = get_protocol_without_rate_constraint(line_topology, demand)
                    if protocol is None:
                        logger.warning("Demand {} could not be satisfied!".format(demand))
                        continue


                    logger.debug("Converting protocol for request {} to task".format(demand))
                    task = convert_protocol_to_task(demand, protocol, slot_size)

                    logger.debug("Scheduling task for request {}".format(demand))

                    scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, line_topology)

                    latency = scheduled_task.c * slot_size
                    achieved_rate = 1 / latency

                    new_rate = select_rate(achieved_rate, slot_size)
                    if new_rate == 0:
                        logger.warning("Could not provide rate for {}".format(demand))
                        continue
                    scheduled_task.p = ceil(1 / new_rate / slot_size)

                    s, d, f, r = demand
                    demand = (s, d, f, new_rate)
                    s, d, f, r = demand

                    asap_dec, alap_dec, shift_dec = decoherence_times
                    logger.info("Results for {}:".format(demand))
                    if not correct:
                        logger.error("Failed to construct valid protocol for {}".format(demand))
                        import pdb
                        pdb.set_trace()
                    elif achieved_rate < r:
                        logger.warning("Failed to satisfy rate for {}, achieved {}".format(demand, achieved_rate))
                        import pdb
                        pdb.set_trace()
                    elif shift_dec > asap_dec or shift_dec > alap_dec:
                        logger.error("Shifted protocol has greater decoherence than ALAP or ASAP for demand {}".format(demand))
                        import pdb
                        pdb.set_trace()
                    else:
                        num_succ += 1
                        logger.info("Successfully created protocol and task for demand (S={}, D={}, F={}, R={}), {}".format(*demand, num_succ))
                        taskset.append(scheduled_task)

                except Exception as err:
                    logger.exception("Error occurred while generating tasks: {}".format(err))
                    import pdb
                    pdb.set_trace()

            import pdb
            pdb.set_trace()
            balance_taskset_resource_utilization(taskset, line_topology[1].nodes)
            logger.info("Demands: {}".format(demands))
            total_rate_dict = defaultdict(int)
            for task in taskset:
                total_rate_dict[1 / task.p] += 1

            taskset_properties["rates"] = total_rate_dict
            network_tasksets.append(taskset)
            network_taskset_properties.append(taskset_properties)

            logger.info("Completed creating taskset {}".format(i))
            # Use all schedulers
            for scheduler_class in network_schedulers:
                try:
                    scheduler = scheduler_class()
                    results_key = type(scheduler).__name__
                    running_taskset = []
                    last_succ_schedule = None

                    logger.info("Scheduling tasks with {}".format(results_key))
                    start = time.time()
                    for task in taskset:
                        # First test the taskset if it is even feasible to schedule
                        test_taskset = running_taskset + [task]
                        if check_resource_utilization(test_taskset) == False:
                            continue

                        schedule = scheduler.schedule_tasks(running_taskset + [task], topology)
                        if schedule:
                            # Record success
                            if all([valid for _, _, valid in schedule]):
                                running_taskset.append(task)
                                logger.info("Running taskset length: {}".format(len(running_taskset)))
                                last_succ_schedule = schedule
                                for sub_taskset, sub_schedule, _ in schedule:
                                    logger.debug("Created schedule for {} demands {}, length={}".format(
                                        len(sub_taskset), [t.name for t in sub_taskset],
                                        max([slot_info[1] for slot_info in sub_schedule])))

                            else:
                                logger.warning("Could not add demand {} with latency {}".format(task.name, task.c*slot_size))

                    end = time.time()
                    logger.info("{} completed scheduling in {}s".format(results_key, end - start))
                    logger.info("{} scheduled {} tasks".format(results_key, len(running_taskset)))

                    rate_dict = defaultdict(int)

                    for task in running_taskset:
                        rate_dict[1/task.p] += 1

                    logger.info("Taskset {} statistics:".format(i))
                    logger.info("Rates: ")

                    num_pairs = 0
                    hyperperiod = get_lcm_for([t.p for t in running_taskset])
                    for rate in sorted(total_rate_dict.keys()):
                        num_pairs += rate_dict[rate] * hyperperiod * rate
                        logger.info("{}: {} / {}".format(rate, rate_dict[rate], total_rate_dict[rate]))

                    total_latency = hyperperiod*slot_size
                    logger.info("Schedule generates {} pairs in {}s".format(num_pairs, total_latency))

                    network_throughput = num_pairs / total_latency
                    logger.info("Network Throughput: {} ebit/s".format(network_throughput))

                    for sub_taskset, sub_schedule, _ in last_succ_schedule:
                        sub_schedule_pairs = 0
                        for task in sub_taskset:
                            sub_schedule_pairs += hyperperiod / task.p
                        logger.info("Sub taskset {}: Num pairs {} Latency {}".format([t.name for t in sub_taskset],
                                                                                     sub_schedule_pairs,
                                                                                     slot_size*max([slot_info[1] for slot_info in sub_schedule])))
                        schedule_and_resource_timelines(sub_taskset, sub_schedule, plot_title=results_key)

                    # Data is taskset_num, number_scheduled_tasks, overall throughput, rate dict
                    # satisfied demands
                    scheduler_results = (i, len(running_taskset), network_throughput, rate_dict, [t.name for t in running_taskset])
                    results[results_key] = scheduler_results

                except Exception as err:
                    logger.exception("Error occurred while scheduling: {}".format(err))

        except Exception as err:
            logger.exception("Unknown error occurred: {}".format(err))

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()