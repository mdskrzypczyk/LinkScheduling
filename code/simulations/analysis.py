import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from os import listdir

font = {'family': 'normal',
        'size': 18}

matplotlib.rc('font', **font)


def get_wcrt_in_slots(schedule, slot_size):
    """
    Obtains the worst-case response time of a schedule
    :param schedule: type list
        List of (start, end, task) describing the schedule
    :param slot_size: type float
        The size of a slot in seconds
    :return: type float
        The worst-case response time observed in the schedule
    """
    task_wcrts = defaultdict(int)
    for s, e, t in schedule:
        name_components = t.name.split("|")
        instance_name = "|".join(name_components[:2])
        task_wcrts[instance_name] = (e - t.a) * slot_size

    original_task_wcrts = defaultdict(int)
    for instance_name, wcrt in task_wcrts.items():
        original_taskname = instance_name.split("|")[0]
        original_task_wcrts[original_taskname] = max(original_task_wcrts[original_taskname], wcrt)

    return original_task_wcrts


def get_start_jitter_in_slots(taskset, schedule, slot_size):
    """
    Obtains the jitter of all tasks in a schedule
    :param taskset: type list
        List of the PeriodicTasks that were used for the schedule
    :param schedule: type list
        List of (start, end, task) describing the schedule
    :param slot_size: type float
        The size of a slot in seconds
    :return: type float
        The worst-case response time observed in the schedule
    """
    periodic_task_starts = defaultdict(list)
    for s, e, t in schedule:
        name_components = t.name.split("|")
        original_taskname = name_components[0]
        if len(name_components) == 3:
            if name_components[2] == "0":
                periodic_task_starts[original_taskname].append(s * slot_size)
        else:
            periodic_task_starts[original_taskname].append(s * slot_size)

    periodic_task_start_jitter = defaultdict(int)
    for periodic_task in taskset:
        task_starts = periodic_task_starts[periodic_task.name]
        change = []
        for s1, s2 in zip(task_starts, task_starts[1:]):
            diff = s2 - s1 - periodic_task.p * slot_size
            change.append(diff)

        if change:
            jitter = np.var(change)
            periodic_task_start_jitter[periodic_task.name] = jitter

    return periodic_task_start_jitter


def load_results_from_files(files):
    """
    Loads simulation results from a list of files
    :param files: type list
        List of filenames to load results from
    :return: type dict
        Dictionary of the results from the files
    """
    results = {}
    for file in files:
        file_results = json.load(open(file))
        results.update(file_results)

    return results


def plot_results(data):
    """
    Plots the results obtained from simulation files
    :param data: type dict
        A dictionary of the simulation results
    :return: None
    """
    entry_key = list(data.keys())[0]
    fidelities = list(data[entry_key].keys())
    schedulers = list(sorted(data[entry_key][fidelities[0]]))
    label_map = {
        "MultipleResourceBlockCEDFScheduler": "RCPSP-CEDF",
        "MultipleResourceNonBlockNPEDFScheduler": "RCPSP-NP-EDF",
        "MultipleResourceBlockNPEDFScheduler": "RCPSP-NP-FPR",
        "UniResourceCEDFScheduler": "PTS-CEDF",
        "UniResourceBlockNPEDFScheduler": "PTS-NP-EDF",
        # "UniResourceConsiderateFixedPointPreemptionBudgetScheduler": "PTS-PB",
        # "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler": "RCPSP-PBS-1s",
        # "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler": "RCPSP-PB-1s"
    }
    schedulers = [sched for sched in schedulers if sched in label_map.keys()]
    for metric in ["throughput", "wcrt", "jitter"]:
        means = defaultdict(lambda: defaultdict(list))
        errs = defaultdict(lambda: defaultdict(float))
        increase = defaultdict(lambda: defaultdict(list))
        errs_increase = defaultdict(lambda: defaultdict(float))
        for sched in schedulers:
            for run_key, run_data in data.items():
                for fidelity in fidelities:
                    try:
                        means[fidelity][sched].append(run_data[fidelity][sched][metric])
                        if metric in ["throughput", "wcrt"]:
                            if "UniResource" in sched:
                                baseline = "UniResourceBlockNPEDFScheduler"
                            else:
                                baseline = "MultipleResourceNonBlockNPEDFScheduler"
                            diff = run_data[fidelity][sched][metric] - run_data[fidelity][baseline][metric]
                            increase[fidelity][sched].append(diff / run_data[fidelity][baseline][metric])
                    except Exception:
                        import pdb
                        pdb.set_trace()

        for sched in schedulers:
            for fidelity in fidelities:
                errs[fidelity][sched] = np.std(means[fidelity][sched])
                means[fidelity][sched] = np.mean(means[fidelity][sched])
                if metric in ["throughput", "wcrt"]:
                    errs_increase[fidelity][sched] = np.std(increase[fidelity][sched])
                    increase[fidelity][sched] = np.mean(increase[fidelity][sched])

        labels = [label_map[sched] for sched in schedulers]
        x = np.arange(len(labels))  # the label locations
        total_width = 0.7       # Width of all bars
        width = total_width / len(means.keys())   # the width of the bars

        fig, ax = plt.subplots()
        offset = (len(means.keys()) - 1) * width / 2
        for sched in schedulers:
            print(sched, [means[fidelity][sched] for fidelity in means.keys()])
            print(sched, [increase[fidelity][sched] for fidelity in increase.keys()])

        for i, fidelity in enumerate(means.keys()):
            ydata = [means[fidelity][sched] for sched in schedulers]
            yerr = [errs[fidelity][sched] for sched in schedulers]
            if metric in ["throughput"]:
                ax.bar(x - offset + i * width, ydata, yerr=yerr, width=width, label="$F$={}".format(fidelity))
            else:
                ax.bar(x - offset + i * width, ydata, width=width, label="$F$={}".format(fidelity))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        metric_to_label = {"throughput": "Throughput", "wcrt": "Worst-case Response Time", "jitter": "Jitter"}
        metric_to_units = {"throughput": "(ebit/s)", "wcrt": "(s)", "jitter": "($s^2$)"}
        ax.set_ylabel("{} {}".format(metric_to_label[metric], metric_to_units[metric]))
        ax.set_title('{} by Scheduler and Fidelity'.format(metric_to_label[metric]))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

        fig.tight_layout()
        plt.draw()

        plt.show()

        if metric in ["throughput", "wcrt"]:
            fig, ax = plt.subplots()
            offset = (len(means.keys()) - 1) * width / 2
            for i, fidelity in enumerate(means.keys()):
                ydata = [increase[fidelity][sched] for sched in schedulers]
                yerr = [errs_increase[fidelity][sched] for sched in schedulers]
                ax.bar(x - offset + i * width, ydata, yerr=yerr, width=width, label="$F$={}".format(fidelity))

            # Add some text for labels, title and custom x-axis tick labels, etc.
            metric_to_label = {"throughput": "Throughput", "wcrt": "Worst-case Response Time", "jitter": "Jitter"}
            metric_to_units = {"throughput": "(ebit/s)", "wcrt": "(s)", "jitter": "($s^2$)"}
            ax.set_ylabel("% Increase".format(metric_to_label[metric], metric_to_units[metric]))
            ax.set_title('{} Increase by Scheduler and Fidelity'.format(metric_to_label[metric]))
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(loc='upper right')

            fig.tight_layout()
            plt.draw()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
            ax.axhline(c="k")
            plt.show()


def plot_pb_results(data):
    """
    Plots preemption budget results
    :param data: type dict
        A dictionary of the simulation results
    :return: None
    """
    entry_key = list(data.keys())[0]
    fidelities = list(data[entry_key].keys())
    label_map = {
        "MultipleResourceNonBlockNPEDFScheduler": "RCPSP-NP-EDF",
        "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler_0.01s": "RCPSP-PB-0.01",
        "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler_0.1s": "RCPSP-PB-0.1",
        "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler_1s": "RCPSP-PB-1",
        "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler_0.01s": "RCPSP-PBS-0.01",
        "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler_0.1s": "RCPSP-PBS-0.1",
        "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler_1s": "RCPSP-PBS-1",
        "UniResourceConsiderateFixedPointPreemptionBudgetScheduler_0.01s": "PTS-PB-0.01",
        "UniResourceConsiderateFixedPointPreemptionBudgetScheduler_0.1s": "PTS-PB-0.1",
        "UniResourceConsiderateFixedPointPreemptionBudgetScheduler_1s": "PTS-PB-1",
        "MultipleResourceBlockNPEDFScheduler": "RCPSP-NP-FPR",
        "UniResourceBlockNPEDFScheduler": "PTS-NP-EDF"
    }
    schedulers = ["MultipleResourceNonBlockNPEDFScheduler",
                  "UniResourceConsiderateFixedPointPreemptionBudgetScheduler_0.01s",
                  "UniResourceConsiderateFixedPointPreemptionBudgetScheduler_0.1s",
                  "UniResourceConsiderateFixedPointPreemptionBudgetScheduler_1s",
                  "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler_0.01s",
                  "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler_0.1s",
                  "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler_1s",
                  "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler_0.01s",
                  "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler_0.1s",
                  "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler_1s"]

    for metric in ["throughput", "wcrt", "jitter"]:
        means = defaultdict(lambda: defaultdict(list))
        errs = defaultdict(lambda: defaultdict(float))
        increase = defaultdict(lambda: defaultdict(list))
        for sched in schedulers:
            for run_key, run_data in data.items():
                for fidelity in fidelities:
                    try:
                        means[fidelity][sched].append(run_data[fidelity][sched][metric])
                        if metric in ["throughput", "wcrt"]:
                            if "UniResource" in sched:
                                baseline = "UniResourceBlockNPEDFScheduler"
                            else:
                                baseline = "MultipleResourceNonBlockNPEDFScheduler"
                            diff = run_data[fidelity][sched][metric] - run_data[fidelity][baseline][metric]
                            increase[fidelity][sched].append(diff / run_data[fidelity][baseline][metric])
                    except Exception:
                        import pdb
                        pdb.set_trace()

        for sched in schedulers:
            for fidelity in fidelities:
                errs[fidelity][sched] = np.std(means[fidelity][sched])
                means[fidelity][sched] = np.mean(means[fidelity][sched])
                if metric in ["throughput", "wcrt"]:
                    increase[fidelity][sched] = np.mean(increase[fidelity][sched])

        for sched in schedulers:
            print(sched, [means[fidelity][sched] for fidelity in means.keys()])

        labels = [label_map[sched] for sched in schedulers]
        x = np.arange(len(labels))  # the label locations
        total_width = 0.7       # Width of all bars
        width = total_width / len(means.keys())   # the width of the bars

        fig, ax = plt.subplots()
        offset = (len(means.keys()) - 1) * width / 2
        for i, fidelity in enumerate(means.keys()):
            ydata = [means[fidelity][sched] for sched in schedulers]
            yerr = [errs[fidelity][sched] for sched in schedulers]
            if metric in ["throughput"]:
                ax.bar(x - offset + i * width, ydata, yerr=yerr, width=width, label="$F$={}".format(fidelity))
            else:
                ax.bar(x - offset + i * width, ydata, width=width, label="$F$={}".format(fidelity))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        metric_to_label = {"throughput": "Throughput", "wcrt": "Worst-case Response Time", "jitter": "Jitter"}
        metric_to_units = {"throughput": "(ebit/s)", "wcrt": "(s)", "jitter": "($s^2$)"}
        ax.set_ylabel("{} {}".format(metric_to_label[metric], metric_to_units[metric]))
        ax.set_title('{} by Scheduler and Fidelity'.format(metric_to_label[metric]))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)

        fig.tight_layout()

        plt.show()

        if metric in ["throughput", "wcrt"]:
            fig, ax = plt.subplots()
            offset = (len(means.keys()) - 1) * width / 2
            for i, fidelity in enumerate(means.keys()):
                ydata = [increase[fidelity][sched] * 100 for sched in schedulers]
                ax.bar(x - offset + i * width, ydata, width=width, label="$F$={}".format(fidelity))

            # Add some text for labels, title and custom x-axis tick labels, etc.
            metric_to_label = {"throughput": "Throughput", "wcrt": "Worst-case Response Time", "jitter": "Jitter"}
            metric_to_units = {"throughput": "(ebit/s)", "wcrt": "(s)", "jitter": "($s^2$)"}
            ax.set_ylabel("% Increase".format(metric_to_label[metric], metric_to_units[metric]))
            ax.set_title('{} Increase by Scheduler and Fidelity'.format(metric_to_label[metric]))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20)
            ax.legend(loc='upper right')

            fig.tight_layout()
            plt.draw()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.23), ncol=4)
            ax.axhline(c="k")
            plt.show()


def plot_load_v_throughput_results(data):
    entry_key = list(data.keys())[0]
    fidelities = list(data[entry_key].keys())
    loads = [str(i) for i in range(1, 21)]
    schedulers = list(sorted(data[entry_key][fidelities[0]][loads[0]]))
    label_map = {
        "MultipleResourceBlockCEDFScheduler": "RCPSP-CEDF",
        "MultipleResourceNonBlockNPEDFScheduler": "RCPSP-NP-EDF",
        "MultipleResourceBlockNPEDFScheduler": "RCPSP-NP-FPR",
        "UniResourceCEDFScheduler": "PTS-CEDF",
        "UniResourceBlockNPEDFScheduler": "PTS-NP-EDF",
        # "UniResourceConsiderateFixedPointPreemptionBudgetScheduler": "PTS-PB",
        # "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler": "RCPSP-PBS-1s",
        # "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler": "RCPSP-PB-1s"
    }
    schedulers = [sched for sched in schedulers if sched in label_map.keys()]
    for fidelity in fidelities:
        means = defaultdict(lambda: defaultdict(list))
        errs = defaultdict(lambda: defaultdict(float))
        for sched in schedulers:
            for run_key, run_data in data.items():
                for load in loads:
                    try:
                        means[load][sched].append(run_data[fidelity][load][sched]["throughput"])
                    except Exception:
                        import pdb
                        pdb.set_trace()

        for sched in schedulers:
            for load in loads:
                errs[load][sched] = np.std(means[load][sched])
                means[load][sched] = np.mean(means[load][sched])

        for sched in schedulers:
            print(sched, [means[load][sched] for load in means.keys()])

        fig, ax = plt.subplots()
        for sched in schedulers:
            ydata = [means[load][sched] for load in loads]
            yerr = [errs[load][sched] for load in loads]
            ax.errorbar(loads, ydata, yerr, label=label_map[sched])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Throughput (ebit/s)")
        ax.set_title('Throughput vs. Network Load')
        ax.set_label("Network Load (ebit/s)")

        # Put a legend below current axis
        ax.legend()

        fig.tight_layout()

        plt.show()


def check_star_results():
    """
    Checks results for the star graph simulations
    :return: None
    """
    files = ["results/star_results/{}".format(file) for file in listdir("results/star_results") if "1c3s" in file]
    results = load_results_from_files(files)
    plot_results(results)


def check_H_results():
    """
    Checks results for the H graph simulations
    :return: None
    """
    files = ["results/H_results/{}".format(file) for file in listdir("results/H_results")]
    files += ["H_results.json"]
    results = load_results_from_files(files)
    plot_results(results)


def check_line_results():
    """
    Checks results for the line graph simulations
    :return: None
    """
    files = ["results/line_results/{}".format(file) for file in listdir("results/line_results")]
    results = load_results_from_files(files)
    plot_results(results)
    
    
def check_symm_results():
    """
    Checks results for the line graph simulations
    :return: None
    """
    files = ["symm_results.json"]
    results = load_results_from_files(files)
    plot_results(results)


def check_pb_results():
    """
    Checks results for the preemption budget simulations
    :return: None
    """
    files = ["results/pb_results/{}".format(file) for file in listdir("results/pb_results")]
    results = load_results_from_files(files)
    plot_pb_results(results)


def check_star_res_results():
    """
    Checks results for the resource allocation simulations
    :return: None
    """
    files_1c3s = ["results/star_results/{}".format(file) for file in listdir("results/star_results") if "1c3s" in file]
    files_1c4s = ["results/star_results/{}".format(file) for file in listdir("results/star_results") if "1c4s" in file]
    files_2c3s = ["results/star_results/{}".format(file) for file in listdir("results/star_results") if "2c3s" in file]
    files_2c4s = ["results/star_results/{}".format(file) for file in listdir("results/star_results") if "2c4s" in file]

    results_1c3s = load_results_from_files(files_1c3s)
    results_1c4s = load_results_from_files(files_1c4s)
    results_2c3s = load_results_from_files(files_2c3s)
    results_2c4s = load_results_from_files(files_2c4s)

    def get_means(data, metric, sched):
        entry_key = list(data.keys())[0]
        fidelities = list(data[entry_key].keys())
        means = defaultdict(lambda: defaultdict(list))
        errs = defaultdict(lambda: defaultdict(float))
        for run_key, run_data in data.items():
            for fidelity in fidelities:
                try:
                    means[fidelity][sched].append(run_data[fidelity][sched][metric])

                except Exception:
                    import pdb
                    pdb.set_trace()

        for fidelity in fidelities:
            errs[fidelity][sched] = np.std(means[fidelity][sched])
            means[fidelity][sched] = np.mean(means[fidelity][sched])

        return means, errs

    scheduler = "MultipleResourceNonBlockNPEDFScheduler"
    for metric in ["throughput", "wcrt", "jitter"]:
        mean_data = {
            "1C-3S": get_means(results_1c3s, metric, scheduler),
            "1C-4S": get_means(results_1c4s, metric, scheduler),
            "2C-3S": get_means(results_2c3s, metric, scheduler),
            "2C-4S": get_means(results_2c4s, metric, scheduler),
        }
        labels = ["1C-4S", "2C-3S", "2C-4S"]
        x = np.arange(len(labels))  # the label locations
        total_width = 0.7  # Width of all bars
        width = total_width / len(mean_data["1C-3S"][0].keys())  # the width of the bars

        fig, ax = plt.subplots()
        offset = (len(mean_data["1C-3S"][0].keys()) - 1) * width / 2
        increase = defaultdict(lambda: defaultdict(float))

        for i, fidelity in enumerate(mean_data["1C-3S"][0].keys()):
            for config in labels:
                mean_config = mean_data[config][0][fidelity][scheduler]
                mean_baseline = mean_data["1C-3S"][0][fidelity][scheduler]
                increase[config][fidelity] = (mean_config - mean_baseline) / mean_baseline

            ydata = [100 * increase[config][fidelity] for config in labels]
            ax.bar(x - offset + i * width, ydata, width=width, label="$F$={}".format(fidelity))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        metric_to_label = {"throughput": "Throughput", "wcrt": "Worst-case Response Time", "jitter": "Jitter"}
        metric_to_units = {"throughput": "(ebit/s)", "wcrt": "(s)", "jitter": "($s^2$)"}
        ax.set_ylabel("% Increase".format(metric_to_label[metric], metric_to_units[metric]))
        ax.set_title('{} Increase by Scheduler and Fidelity'.format(metric_to_label[metric]))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        ax.axhline(c="k")

        fig.tight_layout()
        plt.draw()

        plt.show()


def check_res_results():
    """
    Checks results for the resource to protocol allocation simulations
    :return: None
    """
    files = ["results/resource_results/{}".format(file) for file in listdir("results/resource_results")]
    data = load_results_from_files(files)
    entry_key = list(data.keys())[0]
    fidelities = list(data[entry_key].keys())
    schedulers = list(sorted(data[entry_key][fidelities[0]]))
    label_map = {
        "MultipleResourceBlockCEDFScheduler": "RCPSP-CEDF",
        "MultipleResourceNonBlockNPEDFScheduler": "RCPSP-NP-EDF",
        "MultipleResourceBlockNPEDFScheduler": "RCPSP-NP-FPR",
        # "UniResourceCEDFScheduler": "PTS-CEDF",
        # "UniResourceBlockNPEDFScheduler": "PTS-NP-EDF",
        # "UniResourceConsiderateFixedPointPreemptionBudgetScheduler": "PTS-EDF-LBF",
        "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler": "RCPSP-PBS-1s",
        "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler": "RCPSP-PB-1s"
    }
    for metric in ["throughput", "wcrt", "jitter"]:
        means = defaultdict(lambda: defaultdict(list))
        errs = defaultdict(lambda: defaultdict(float))
        increase = defaultdict(lambda: defaultdict(list))
        for sched in schedulers:
            for run_key, run_data in data.items():
                for fidelity in fidelities:
                    try:
                        means[fidelity][sched].append(run_data[fidelity][sched][metric])
                        if metric in ["throughput", "wcrt"]:
                            s_name, resource_configuration = sched.split("_")
                            baseline = s_name + "_full"
                            diff = run_data[fidelity][sched][metric] - run_data[fidelity][baseline][metric]
                            increase[fidelity][s_name].append(diff / run_data[fidelity][baseline][metric])
                    except Exception:
                        import pdb
                        pdb.set_trace()

        for sched in schedulers:
            for fidelity in fidelities:
                errs[fidelity][sched] = np.std(means[fidelity][sched])
                means[fidelity][sched] = np.mean(means[fidelity][sched])

        if metric in ["throughput", "wcrt"]:
            for sched in increase[fidelities[0]].keys():
                for fidelity in fidelities:
                    increase[fidelity][sched] = np.mean(increase[fidelity][sched])

            labels = [label_map[sched] for sched in label_map.keys()]
            x = np.arange(len(labels))  # the label locations
            total_width = 0.7  # Width of all bars
            width = total_width / len(means.keys())  # the width of the bars

            fig, ax = plt.subplots()
            offset = (len(means.keys()) - 1) * width / 2
            import pdb
            pdb.set_trace()
            for i, fidelity in enumerate(means.keys()):
                ydata = [increase[fidelity][sched] * 100 for sched in label_map.keys()]
                ax.bar(x - offset + i * width, ydata, width=width, label="$F$={}".format(fidelity))

            # Add some text for labels, title and custom x-axis tick labels, etc.
            metric_to_label = {"throughput": "Throughput", "wcrt": "Worst-case Response Time", "jitter": "Jitter"}
            metric_to_units = {"throughput": "(ebit/s)", "wcrt": "(s)", "jitter": "($s^2$)"}
            ax.set_ylabel("% Increase".format(metric_to_label[metric], metric_to_units[metric]))
            ax.set_title('{} Increase by Scheduler and Fidelity'.format(metric_to_label[metric]))
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(loc='upper right')

            fig.tight_layout()
            plt.draw()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
            ax.axhline(c="k")
            plt.show()


def check_load_results():
    files = ["load_results.json"]
    results = load_results_from_files(files)
    plot_load_v_throughput_results(results)
