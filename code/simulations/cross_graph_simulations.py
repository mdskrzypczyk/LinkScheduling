import json
import time
from collections import defaultdict
from jobscheduling.task import get_lcm_for, get_gcd_for
from jobscheduling.topology import gen_topologies, gen_plus_topology, gen_grid_topology, gen_surfnet_topology

def sample_sim():
    center_resource_configs = [(1, 1)] #[(2, 4)]#, (2, 3), (1, 4), (2, 4)]
    end_node_resources = (1, 1) #(1, 3)
    fidelities = [0.6, 0.7, 0.8, 0.9] # [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    num_tasksets = 1
    taskset_sizes = [50, 40, 30, 10]
    slot_size = 0.05
    schedulers = get_schedulers()

    all_data = {}
    for center_resources in center_resource_configs:
        resource_config_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # topology = gen_plus_topology(5, end_node_resources=end_node_resources, center_resources=center_resources,
        #                              link_distance=5)
        topology = gen_grid_topology(9, end_node_resources=end_node_resources, repeater_resources=center_resources,
                                     link_distance=5)
        network_resources = get_network_resources(topology)
        print("Running center config {}".format(center_resources))
        for fidelity, num_tasks in zip(fidelities, taskset_sizes):
            print("Running fidelity {}".format(fidelity))
            for num_taskset in range(num_tasksets):
                taskset = get_taskset(num_tasks, fidelity, topology, slot_size)
                print("Running taskset {}".format(num_taskset))

                total_rate_dict = defaultdict(int)
                for task in taskset:
                    total_rate_dict[1 / task.p] += 1

                for scheduler_class in schedulers:
                    try:
                        scheduler = scheduler_class()
                        results_key = type(scheduler).__name__
                        print("Running scheduler {}".format(results_key))
                        running_taskset = []
                        last_succ_schedule = None

                        logger.info("Scheduling tasks with {}".format(results_key))
                        start = time.time()
                        for task in taskset:
                            # First test the taskset if it is even feasible to schedule
                            test_taskset = running_taskset + [task]
                            if check_resource_utilization(test_taskset, network_resources) == False:
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
                                    logger.warning(
                                        "Could not add demand {} with latency {}".format(task.name, task.c * slot_size))

                        end = time.time()
                        logger.info("{} completed scheduling in {}s".format(results_key, end - start))
                        logger.info("{} scheduled {} tasks".format(results_key, len(running_taskset)))

                        rate_dict = defaultdict(int)

                        for task in running_taskset:
                            rate_dict[1 / task.p] += 1

                        logger.info("Taskset {} statistics:".format(num_taskset))
                        logger.info("Rates: ")

                        num_pairs = 0
                        hyperperiod = get_lcm_for([t.p for t in running_taskset])
                        for rate in sorted(total_rate_dict.keys()):
                            num_pairs += rate_dict[rate] * hyperperiod * rate
                            logger.info("{}: {} / {}".format(rate, rate_dict[rate], total_rate_dict[rate]))

                        total_latency = hyperperiod * slot_size
                        logger.info("Schedule generates {} pairs in {}s".format(num_pairs, total_latency))

                        network_throughput = num_pairs / total_latency
                        logger.info("Network Throughput: {} ebit/s".format(network_throughput))

                        task_wcrts = {}
                        task_jitters = {}
                        for sub_taskset, sub_schedule, _ in last_succ_schedule:
                            subtask_wcrts = get_wcrt_in_slots(sub_schedule, slot_size)
                            subtask_jitters = get_start_jitter_in_slots(running_taskset, sub_schedule, slot_size)
                            task_wcrts.update(subtask_wcrts)
                            task_jitters.update(subtask_jitters)
                            # schedule_and_resource_timelines(sub_taskset, sub_schedule)

                        num_demands = sum(rate_dict.values())
                        resource_config_data[results_key][fidelity]["throughput"].append(network_throughput)
                        resource_config_data[results_key][fidelity]["wcrt"].append(max(task_wcrts.values()))
                        resource_config_data[results_key][fidelity]["jitter"].append(max(task_jitters.values()))
                        resource_config_data[results_key][fidelity]["num_demands"].append(num_demands)

                    except Exception as err:
                        logger.exception("Error occurred while scheduling: {}".format(err))

        all_data[center_resources] = resource_config_data

    final_data = {}
    for res_conf, res_conf_data in all_data.items():
        final_res_conf_key = str(res_conf)
        final_res_conf_data = {}
        for fidelity, fidelity_data in res_conf_data.items():
            final_fidelity_data = {}
            for sched_name, sched_data in fidelity_data.items():
                final_sched_data = {}
                for metric_name, metric_data in sched_data.items():
                    average_metric_data = sum(metric_data) / len(metric_data)
                    final_sched_data[metric_name] = average_metric_data

                final_fidelity_data[sched_name] = final_sched_data

            final_res_conf_data[fidelity] = final_fidelity_data

        final_data[final_res_conf_key] = final_res_conf_data

    json.dump(final_data, open("out.json", "w"), sort_keys=True, indent=4)
    plot_results(final_data)