import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from os import listdir
from scipy.stats import sem

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
        try:
            file_results = json.load(open(file))
        except Exception as err:
            print("Failed to read {}, {}".format(file, err))
            file_contents = open(file).read()
            if "\n}\n   " in file_contents:
                print("Fixing")
                file_contents = file_contents.replace("\n}\n   ", ",\n   ")
                with open(file, "w") as f:
                    f.write(file_contents)
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
    print("Constructing plots from {} datapoints".format(len(data.keys())))
    entry_key = list(data.keys())[0]
    fidelities = list(data[entry_key].keys())
    schedulers = list(sorted(data[entry_key][fidelities[0]]))
    label_map = {
        # "MultipleResourceBlockCEDFScheduler": "RCPSP-CEDF",
        "MultipleResourceNonBlockNPEDFScheduler": "RCPSP-NP-EDF",
        # "MultipleResourceBlockNPEDFScheduler": "RCPSP-NP-FPR",
        # "UniResourceCEDFScheduler": "PTS-CEDF",
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
                            increase[fidelity][sched].append(100 * diff / run_data[fidelity][baseline][metric])
                    except Exception:
                        continue

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
            print(sched, [means[fidelity][sched] for fidelity in sorted(means.keys())])
            print(sched, [errs[fidelity][sched] for fidelity in sorted(means.keys())])
            print(sched, [increase[fidelity][sched] for fidelity in sorted(increase.keys())])

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
                            increase[fidelity][sched].append(100 * diff / run_data[fidelity][baseline][metric])
                    except Exception:
                        continue

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
                ydata = [increase[fidelity][sched] for sched in schedulers]
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
    """
    Plots the achieved throughput as a function of the input load
    :param data: type dict
        A dictionary of the simulation results
    :return: None
    """
    entry_key = list(data.keys())[0]
    fidelities = list(data[entry_key].keys())
    fidelities = ['0.55', '0.65', '0.75', '0.85']
    # fidelities = ['0.7', '0.8']
    # fidelities = ['0.995', '0.997']
    # fidelities = ['0.55', '0.85']
    loads = [str(i) for i in [0.5, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]]
    load_key = list(data[entry_key][fidelities[0]].keys())[0]
    schedulers = list(sorted(data[entry_key][fidelities[0]][load_key]))
    label_map = {
        # "MultipleResourceBlockCEDFScheduler": "RCPSP-CEDF",
        "MultipleResourceNonBlockNPEDFScheduler": "RCPSP-NP-EDF",
        "MultipleResourceBlockNPEDFScheduler": "RCPSP-NP-FPR",
        # "UniResourceCEDFScheduler": "PTS-CEDF",
        "UniResourceBlockNPEDFScheduler": "PTS-NP-EDF",
        # "UniResourceConsiderateFixedPointPreemptionBudgetScheduler": "PTS-PB",
        # "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler": "RCPSP-PBS-1s",
        # "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler": "RCPSP-PB-1s"
    }
    schedulers = [sched for sched in schedulers if sched in label_map.keys()]

    matplotlib.rc('font', **font)
    fig, axs = plt.subplots(1, len(fidelities))
    all_axes = axs # [ax for ax_list in axs for ax in ax_list]
    for i, (fidelity, ax) in enumerate(zip(fidelities, all_axes)):
        count = defaultdict(lambda:defaultdict(int))
        means = defaultdict(lambda: defaultdict(list))
        errs = defaultdict(lambda: defaultdict(float))
        for sched in schedulers:
            for run_key, run_data in data.items():
                for load in loads:
                    if fidelity in run_data.keys() and load in run_data[fidelity].keys():
                        try:
                            achieved_throughput = run_data[fidelity][load][sched]["throughput"]
                            all_demands = run_data[fidelity][load][sched]["satisfied_demands"]
                            all_demands += run_data[fidelity][load][sched]["unsatisfied_demands"]
                            true_load = sum([float(demand.split("R=")[1].split(", ")[0]) for demand in all_demands])
                            if true_load >= float(load):
                                means[true_load][sched].append(achieved_throughput)
                                count[true_load][sched] += 1
                        except Exception as err:
                            continue

        for sched in schedulers:
            for load in sorted(means.keys()):
                errs[load][sched] = np.std(means[load][sched])
                means[load][sched] = np.mean(means[load][sched])

        for sched in schedulers:
            print(sched, [(load, count[load][sched], means[load][sched]) for load in sorted(means.keys())])

        fmt_map = {
            "MultipleResourceBlockCEDFScheduler": ("-", '0.7'),
            "MultipleResourceNonBlockNPEDFScheduler": (":", '0.4'),
            "MultipleResourceBlockNPEDFScheduler": ("-.", '0.1'),
            "UniResourceCEDFScheduler": ("--", '0.1'),
            "UniResourceBlockNPEDFScheduler": ("-", '0.4')
        }
        for sched in schedulers:
            fmt, c = fmt_map[sched]
            xdata = list(sorted(means.keys()))
            ydata = [means[load][sched] for load in xdata]
            yerr = [errs[load][sched] for load in xdata]
            (_, caps, _) = ax.errorbar(xdata, ydata, yerr, fmt=fmt, color=c, fillstyle='none', label=label_map[sched], markersize=3, capsize=2)
            for cap in caps:
                cap.set_markeredgewidth(2)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel("Load (ebit/s)", fontsize=14)
        if i in [0, 4]:
            ax.set_ylabel("Throughput (ebit/s)", fontsize=14)
        ax.set_title("$F={}$".format(fidelity), fontsize=14)
        ax.tick_params(axis='both', labelsize=14)

    # fig.delaxes(all_axes[-1])
    handles, labels = all_axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1, 0.35), loc='lower right', fontsize=10)

    def on_resize(event):
        fig.tight_layout()
        fig.canvas.draw()

    fig.canvas.mpl_connect('resize_event', on_resize)
    plt.show()


def check_star_results():
    """
    Checks throughput results for the star graph simulations
    :return: None
    """
    files = ["results/star_results/{}".format(file) for file in listdir("results/star_results") if "1c3s" in file]
    files += ["remote_results/star_results/{}".format(file) for file in listdir("remote_results/star_results")]
    results = load_results_from_files(files)
    plot_results(results)


def check_star_throughputs():
    """
    Checks throughput CDFs for the star graph simulations
    :return: None
    """
    files = ["results/star_results/{}".format(file) for file in listdir("results/star_results") if "1c3s" in file]
    files += ["remote_results/star_results/{}".format(file) for file in listdir("remote_results/star_results")]
    results = load_results_from_files(files)
    plot_achieved_throughput(results)


def check_star_wcrts():
    """
    Checks WCRTs for the star graph simulations
    :return: None
    """
    files = ["results/star_results/{}".format(file) for file in listdir("results/star_results") if "1c3s" in file]
    files += ["remote_results/star_results/{}".format(file) for file in listdir("remote_results/star_results")]
    results = load_results_from_files(files)
    plot_achieved_wcrts(results)


def check_star_jitters():
    """
    Checks jitter for the star graph simulations
    :return: None
    """
    files = ["results/star_results/{}".format(file) for file in listdir("results/star_results") if "1c3s" in file]
    files += ["remote_results/star_results/{}".format(file) for file in listdir("remote_results/star_results")]
    results = load_results_from_files(files)
    plot_achieved_jitter(results)


def check_H_results():
    """
    Checks throughput results for the H graph simulations
    :return: None
    """
    files = ["results/H_results/{}".format(file) for file in listdir("results/H_results")]
    files += ["remote_results/H_results.json"]
    results = load_results_from_files(files)
    plot_results(results)


def check_H_throughputs():
    """
    Checks throughput CDFs for the H graph simulations
    :return: None
    """
    files = ["results/H_results/{}".format(file) for file in listdir("results/H_results")]
    files += ["remote_results/H_results.json"]
    results = load_results_from_files(files)
    plot_achieved_throughput(results)


def check_H_wcrts():
    """
    Checks WCRT results for the H graph simulations
    :return: None
    """
    files = ["results/H_results/{}".format(file) for file in listdir("results/H_results")]
    files += ["remote_results/H_results.json"]
    results = load_results_from_files(files)
    plot_achieved_wcrts(results)


def check_H_jitters():
    """
    Checks jitter results for the H graph simulations
    :return: None
    """
    files = ["results/H_results/{}".format(file) for file in listdir("results/H_results")]
    files += ["remote_results/H_results.json"]
    results = load_results_from_files(files)
    plot_achieved_jitter(results)


def check_line_results():
    """
    Checks throughput results for the line graph simulations
    :return: None
    """
    files = ["results/line_results/{}".format(file) for file in listdir("results/line_results")]
    files += ["remote_results/line_results/{}".format(file) for file in listdir("remote_results/line_results")]
    results = load_results_from_files(files)
    plot_results(results)


def check_line_throughputs():
    """
    Checks throughput CDFs for the line graph simulations
    :return: None
    """
    files = ["results/line_results/{}".format(file) for file in listdir("results/line_results")]
    files += ["remote_results/line_results/{}".format(file) for file in listdir("remote_results/line_results")]
    results = load_results_from_files(files)
    plot_achieved_throughput(results)


def check_line_wcrts():
    """
    Checks WCRT results for the line graph simulations
    :return: None
    """
    files = ["results/line_results/{}".format(file) for file in listdir("results/line_results")]
    files += ["remote_results/line_results/{}".format(file) for file in listdir("remote_results/line_results")]
    results = load_results_from_files(files)
    plot_achieved_wcrts(results)


def check_line_jitters():
    """
    Checks jitter results for the line graph simulations
    :return: None
    """
    files = ["results/line_results/{}".format(file) for file in listdir("results/line_results")]
    files += ["remote_results/line_results/{}".format(file) for file in listdir("remote_results/line_results")]
    results = load_results_from_files(files)
    plot_achieved_jitter(results)
    
    
def check_symm_results():
    """
    Checks results for the line graph simulations
    :return: None
    """
    files = ["results/symm_results/{}".format(file) for file in listdir("results/symm_results/")]
    files += ["remote_results/symm_results.json"]
    results = load_results_from_files(files)
    print("Constructing symm simulation plots from {} datapoints".format(len(results.keys())))
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
                    continue

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


def plot_achieved_throughput(data):
    """
    Plots the results obtained from simulation files
    :param data: type dict
        A dictionary of the simulation results
    :return: None
    """
    entry_key = list(data.keys())[0]
    fidelities = list(data[entry_key].keys())
    # fidelities = ["0.7", "0.8"]
    # fidelities = ["0.55", "0.65", "0.75", "0.85"]
    fidelities = ['0.995', '0.997']
    # fidelities = ['0.55', '0.85']
    schedulers = list(sorted(data[entry_key][fidelities[0]]))
    label_map = {
        # "MultipleResourceBlockCEDFScheduler": "RCPSP-CEDF",
        "MultipleResourceNonBlockNPEDFScheduler": "RCPSP-NP-EDF",
        "MultipleResourceBlockNPEDFScheduler": "RCPSP-NP-FPR",
        # "UniResourceCEDFScheduler": "PTS-CEDF",
        "UniResourceBlockNPEDFScheduler": "PTS-NP-EDF",
        # "UniResourceConsiderateFixedPointPreemptionBudgetScheduler": "PTS-PB",
        # "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler": "RCPSP-PBS-1s",
        # "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler": "RCPSP-PB-1s"
    }
    fmt_map = {
        "MultipleResourceBlockCEDFScheduler": ("-", '0.7'),
        "MultipleResourceNonBlockNPEDFScheduler": (":", '0.4'),
        "MultipleResourceBlockNPEDFScheduler": ("-.", '0.1'),
        "UniResourceCEDFScheduler": ("--", '0.1'),
        "UniResourceBlockNPEDFScheduler": ("-", '0.4')
    }
    schedulers = [sched for sched in schedulers if sched in label_map.keys()]
    for metric in ["throughput"]:
        achieved_throughput = defaultdict(lambda: defaultdict(list))
        for sched in schedulers:
            for run_key, run_data in data.items():
                for fidelity in fidelities:
                    try:
                        achieved_throughput[fidelity][sched].append(run_data[fidelity][sched][metric])
                    except Exception:
                        pass

        matplotlib.rc('font', **font)
        fig, axs = plt.subplots(1, len(fidelities))
        all_axes = axs  # [ax for ax_list in axs for ax in ax_list]
        for i, (fidelity, ax) in enumerate(zip(fidelities, all_axes)):
            for sched in schedulers:
                fmt, c = fmt_map[sched]
                throughputs = list(sorted(achieved_throughput[fidelity][sched] + [0]))
                throughput_counts = defaultdict(int)
                for throughput in throughputs:
                    throughput_counts[throughput] += 1
                xdata = list(sorted(throughput_counts.keys()))
                ydata = [throughput_counts[x] / len(throughputs) for x in xdata]
                ax.hist(throughputs, len(xdata), linestyle=fmt, color=c, density=True, histtype='step', cumulative=True, label=label_map[sched])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_xlabel("Throughput (ebit/s)", fontsize=12)
            ax.set_ylabel("", fontsize=12)
            ax.set_title('$F={}$'.format(fidelity), fontsize=14)

        # Put a legend below current axis
        handles, labels = all_axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.35), loc='lower right', fontsize=10)

        fig.tight_layout()
        plt.draw()

        def on_resize(event):
            fig.tight_layout()
            fig.canvas.draw()

        fig.canvas.mpl_connect('resize_event', on_resize)

        plt.show()


def plot_achieved_wcrts(data):
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
        # "MultipleResourceBlockCEDFScheduler": "RCPSP-CEDF",
        "MultipleResourceNonBlockNPEDFScheduler": "RCPSP-NP-EDF",
        "MultipleResourceBlockNPEDFScheduler": "RCPSP-NP-FPR",
        # "UniResourceCEDFScheduler": "PTS-CEDF",
        "UniResourceBlockNPEDFScheduler": "PTS-NP-EDF",
        # "UniResourceConsiderateFixedPointPreemptionBudgetScheduler": "PTS-PB",
        # "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler": "RCPSP-PBS-1s",
        # "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler": "RCPSP-PB-1s"
    }
    schedulers = [sched for sched in schedulers if sched in label_map.keys()]
    for metric in ["wcrt"]:
        achieved_throughput = defaultdict(lambda: defaultdict(list))
        for sched in schedulers:
            for run_key, run_data in data.items():
                for fidelity in fidelities:
                    try:
                        achieved_throughput[fidelity][sched].append(run_data[fidelity][sched][metric])
                    except Exception:
                        pass

        for i, fidelity in enumerate(achieved_throughput.keys()):
            fig, ax = plt.subplots()
            for sched in schedulers:
                throughputs = achieved_throughput[fidelity][sched]
                throughput_counts = defaultdict(int)
                for throughput in throughputs:
                    throughput_counts[throughput] += 1
                xdata = list(sorted(throughput_counts.keys()))
                ydata = [throughput_counts[x] / len(throughputs) for x in xdata]
                ax.hist(throughputs, len(xdata), density=True, histtype='step', cumulative=True, label=label_map[sched])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_xlabel("Worst-case Response Time (s)")
            ax.set_ylabel("")
            ax.set_title('CDF of Worst-case Response Time for F={}'.format(fidelity))

            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

            fig.tight_layout()
            plt.draw()

            plt.show()


def plot_achieved_jitter(data):
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
    for metric in ["jitter"]:
        achieved_throughput = defaultdict(lambda: defaultdict(list))
        for sched in schedulers:
            for run_key, run_data in data.items():
                for fidelity in fidelities:
                    try:
                        achieved_throughput[fidelity][sched].append(run_data[fidelity][sched][metric])
                    except Exception:
                        pass

        for i, fidelity in enumerate(achieved_throughput.keys()):
            fig, ax = plt.subplots()
            for sched in schedulers:
                throughputs = achieved_throughput[fidelity][sched]
                throughput_counts = defaultdict(int)
                for throughput in throughputs:
                    throughput_counts[throughput] += 1
                xdata = list(sorted(throughput_counts.keys()))
                ydata = [throughput_counts[x] / len(throughputs) for x in xdata]
                ax.hist(throughputs, len(xdata), density=True, histtype='step', cumulative=True, label=label_map[sched])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_xlabel("Jitter (s^2)")
            ax.set_ylabel("")
            ax.set_title('CDF of Jitter for F={}'.format(fidelity))

            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

            fig.tight_layout()
            plt.draw()

            plt.show()


def plot_throughput_wcrt_hist2d(data):
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

    throughput_wcrt_data = defaultdict(lambda: defaultdict(list))
    for sched in schedulers:
        for run_key, run_data in data.items():
            for fidelity in run_data.keys():
                try:
                    throughput_wcrt_data[fidelity][sched].append((run_data[fidelity][sched]["throughput"],
                                                                  run_data[fidelity][sched]["wcrt"]))
                except Exception:
                    pass

    for i, fidelity in enumerate(throughput_wcrt_data.keys()):
        fig, axs = plt.subplots(2, 3)
        all_axes = [ax for ax_list in axs for ax in ax_list]
        for sched, ax in zip(schedulers, all_axes):
            data = throughput_wcrt_data[fidelity][sched]
            xdata = [d[0] for d in data]
            ydata = [d[1] for d in data]
            ax.hist2d(xdata, ydata, [40, 20], [[0, 1 + max(xdata)], [0, 1 + max(ydata)]])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_xlabel("Throughput (ebit/s)")
            ax.set_ylabel("WCRT (s)")
            ax.set_title("{}".format(label_map[sched]))

        # fig.set_title('Throughput vs Jitter for F={}'.format(fidelity))

        # Put a legend below current axis
        # fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        fig.tight_layout()
        plt.draw()

        def on_resize(event):
            fig.tight_layout()
            fig.canvas.draw()

        fig.canvas.mpl_connect('resize_event', on_resize)
        import pdb
        pdb.set_trace()
        plt.show()


def plot_throughput_jitter_hist2d(data):
    """
    Plots the results obtained from simulation files
    :param data: type dict
        A dictionary of the simulation results
    :return: None
    """
    entry_key = list(data.keys())[0]
    fidelities = list(data[entry_key].keys())
    # fidelities = ['0.995', '0.997']
    # fidelities = ['0.55', '0.85']
    schedulers = list(sorted(data[entry_key][fidelities[0]]))
    label_map = {
        # "MultipleResourceBlockCEDFScheduler": "RCPSP-CEDF",
        "MultipleResourceNonBlockNPEDFScheduler": "RCPSP-NP-EDF",
        "MultipleResourceBlockNPEDFScheduler": "RCPSP-NP-FPR",
        # "UniResourceCEDFScheduler": "PTS-CEDF",
        "UniResourceBlockNPEDFScheduler": "PTS-NP-EDF",
        # "UniResourceConsiderateFixedPointPreemptionBudgetScheduler": "PTS-PB",
        # "MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler": "RCPSP-PBS-1s",
        # "MultipleResourceConsiderateSegmentPreemptionBudgetScheduler": "RCPSP-PB-1s"
    }
    schedulers = [sched for sched in schedulers if sched in label_map.keys()]

    throughput_wcrt_data = defaultdict(lambda: defaultdict(list))
    for sched in schedulers:
        for run_key, run_data in data.items():
            for fidelity in run_data.keys():
                try:
                    throughput_wcrt_data[fidelity][sched].append((run_data[fidelity][sched]["throughput"],
                                                                  run_data[fidelity][sched]["jitter"]))
                except Exception:
                    continue

    font = {'family': 'normal',
            'size': 10}

    matplotlib.rc('font', **font)
    fidelities = list(sorted(throughput_wcrt_data.keys()))
    for i, fidelity in enumerate(fidelities):
        fig, axs = plt.subplots(1, len(schedulers))
        all_axes = axs  # [ax for ax in axs[i]]
        for sched, ax in zip(schedulers, all_axes):
            data = throughput_wcrt_data[fidelity][sched]
            xdata = [d[0] for d in data]
            ydata = [d[1] for d in data]
            weights = np.ones_like(xdata) / float(len(xdata))
            h = ax.hist2d(xdata, ydata, bins=30, norm=matplotlib.colors.LogNorm(), cmap=plt.cm.Greys)
            fig.colorbar(h[3], ax=ax)
            ax.set_xlabel("Throughput (ebit/s)", fontsize=12)
            ax.set_title("{}".format(label_map[sched]), fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            ax.set_ylim(0, 0.3)

        fig.tight_layout()

        # Add ylabel text
        leftmost_label = "$F={}$\nJitter ($s^2$)".format(fidelity)
        leftmost_offset = all_axes[0].yaxis.get_offset_text().get_text()
        if leftmost_offset:
            leftmost_label += "\n{}".format(leftmost_offset)
        all_axes[0].set_ylabel(leftmost_label, fontsize=12)

        for ax in all_axes[1:]:
            offset = ax.yaxis.get_offset_text().get_text()
            if offset:
                ax.yaxis.set_label_text(offset)
                ax.yaxis.offsetText.set_visible(False)

        plt.draw()

        def on_resize(event):
            fig.tight_layout()
            fig.canvas.draw()

        fig.canvas.mpl_connect('resize_event', on_resize)

        plt.show()


def plot_scheduler_rankings(data):
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
    for metric in ["throughput"]:
        scheduler_rankings = defaultdict(lambda: defaultdict(float))
        total_runs = len(data.items())
        for run_key, run_data in data.items():
            for fidelity in fidelities:
                try:
                    sched_performance = [(run_data[fidelity][sched][metric], sched) for sched in schedulers]
                    sched_performance = list(sorted(sched_performance, key=lambda d: -d[0]))
                    best_scheduler = sched_performance[0][1]
                    scheduler_rankings[fidelity][best_scheduler] += 1
                except Exception:
                    continue

        for sched in schedulers:
            for fidelity in fidelities:
                scheduler_rankings[fidelity][sched] /= total_runs

        labels = [label_map[sched] for sched in schedulers]
        x = np.arange(len(labels))  # the label locations
        total_width = 0.7       # Width of all bars
        width = total_width / len(scheduler_rankings.keys())   # the width of the bars

        fig, ax = plt.subplots()
        offset = (len(scheduler_rankings.keys()) - 1) * width / 2
        for sched in schedulers:
            print(sched, [scheduler_rankings[fidelity][sched] for fidelity in sorted(scheduler_rankings.keys())])

        for i, fidelity in enumerate(scheduler_rankings.keys()):
            ydata = [100 * scheduler_rankings[fidelity][sched] for sched in schedulers]
            ax.bar(x - offset + i * width, ydata, width=width, label="$F$={}".format(fidelity))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Percent")
        ax.set_title('Percent of Runs Highest Throughput by Scheduler and Fidelity')
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
        # "UniResourceCEDFSplot_resultscheduler": "PTS-CEDF",
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
                        continue

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


def check_symm_load_results():
    """
    Checks load v throughput results for the symm graph simulations
    :return: None
    """
    # files = ["remote_results/load_results.json"] + ["remote_results/load_results{}.json".format(i) for i in range(2, 11)]
    # files += ["remote_results/load_results_high_loads.json"] + ["remote_results/load_results_high_loads_{}.json".format(i) for i in range(2, 5)]
    # files += ["results/load_symm_results/load_results.json"] + ["results/load_symm_results/load_results{}.json".format(i) for i in range(2, 7)]
    result_dir = 'new_results/symm_results/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    results = load_results_from_files(files)
    print("Constructing load simulation results from {} datapoints".format(len(results.keys())))
    plot_load_v_throughput_results(results)


def get_entries_for_load(data, load):
    """
    Unpacks the data from the load controlled simulations
    :param data: type dict
        A dictionary of the simulation results
    :param data: type str
        The load to unpack results for
    :return: None
    """
    new_data = defaultdict(dict)
    for run_key, run_data in data.items():
        for fid_key, fid_data in run_data.items():
            if fid_data.get(load):
                new_data[run_key][fid_key] = fid_data[load]
    return new_data


def check_symm_throughput_cdf():
    """
    Checks throughput CDFs based on load for the line graph simulations
    :return: None
    """
    # files = ["remote_results/load_results.json"] + ["remote_results/load_results{}.json".format(i) for i in
    #                                                 range(2, 11)]
    # files += ["remote_results/load_results_high_loads.json"] + [
    #     "remote_results/load_results_high_loads_{}.json".format(i) for i in range(2, 5)]
    # files += ["results/load_symm_results/load_results.json"] + [
    #     "results/load_symm_results/load_results{}.json".format(i) for i in range(2, 7)]
    result_dir = 'published_results/symmetric_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    loaded_results = load_results_from_files(files)
    for load in [str(i) for i in range(5, 55)]:
        import pdb
        pdb.set_trace()
        results = get_entries_for_load(loaded_results, load)
        print("Constructing symm simulation plots from {} datapoints".format(len(results.keys())))
        plot_achieved_throughput(results)


def check_symm_wcrts():
    """
    Plots the achieved wcrt for the symm topology
    :return: None
    """
    files = ["results/symm_results/symm_results.json", "remote_results/symm_results.json"]
    results = load_results_from_files(files)
    plot_achieved_wcrts(results)


def check_symm_jitters():
    """
    Plots the achieved jitter for the symm topology
    :return: None
    """
    files = ["results/symm_results/symm_results.json", "remote_results/symm_results.json"]
    results = load_results_from_files(files)
    plot_achieved_jitter(results)


def check_symm_throughput_wcrt():
    """
    Plots a 2D histogram of achieved throughput/wcrt pairs for the symm topology
    :return: None
    """
    # files = ["results/symm_results/symm_results.json", "remote_results/symm_results.json"]
    result_dir = 'published_results/symmetric_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    results = load_results_from_files(files)
    plot_throughput_wcrt_hist2d(results)


def check_symm_throughput_jitter():
    """
    Plots a 2D histogram of achieved throughput/jitter pairs based on load for the symm topology
    :return: None
    """
    # files = ["results/symm_results/symm_results.json", "remote_results/symm_results.json"]
    # results = load_results_from_files(files)
    # files = ["remote_results/load_results.json"] + ["remote_results/load_results{}.json".format(i) for i in
    #                                                 range(2, 11)]
    # files += ["remote_results/load_results_high_loads.json"] + [
    #     "remote_results/load_results_high_loads_{}.json".format(i) for i in range(2, 5)]
    # files += ["results/load_symm_results/load_results.json"] + [
    #     "results/load_symm_results/load_results{}.json".format(i) for i in range(2, 7)]
    result_dir = 'published_results/symmetric_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    loaded_results = load_results_from_files(files)
    for load in [str(i) for i in range(5, 55)]:
        import pdb
        pdb.set_trace()
        results = get_entries_for_load(loaded_results, load)
        print("Constructing symm simulation plots from {} datapoints".format(len(results.keys())))
        plot_throughput_jitter_hist2d(results)


def check_star_load_results():
    """
    Checks the load v throughput results from the star topology
    :return: None
    """
    # files = ["remote_results/star_load_results{}.json".format(i) for i in [0, 2, 3, 4, 5]]
    result_dir = 'published_results/star_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    results = load_results_from_files(files)
    print("Constructing star load simulation results from {} datapoints".format(len(results.keys())))
    plot_load_v_throughput_results(results)


def check_star_throughput_cdf():
    """
   Checks throughput CDFs based on load for the line graph simulations
   :return: None
   """
    # files = ["remote_results/star_load_results{}.json".format(i) for i in [0, 2, 3, 4, 5]]
    result_dir = 'published_results/star_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    loaded_results = load_results_from_files(files)
    for load in [str(i) for i in range(5, 55)]:
        import pdb
        pdb.set_trace()
        results = get_entries_for_load(loaded_results, load)
        print("Constructing star simulation plots from {} datapoints".format(len(results.keys())))
        plot_achieved_throughput(results)


def check_star_throughput_jitter():
    """
    Plots a 2D histogram of achieved throughput/jitter pairs based on load for the star topology
    :return: None
    """
    # files = ["remote_results/star_load_results{}.json".format(i) for i in [0, 2, 3, 4, 5]]
    result_dir = 'published_results/star_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    loaded_results = load_results_from_files(files)
    for load in [str(i) for i in range(5, 55)]:
        import pdb
        pdb.set_trace()
        results = get_entries_for_load(loaded_results, load)
        print("Constructing star simulation plots from {} datapoints".format(len(results.keys())))
        plot_throughput_jitter_hist2d(get_entries_for_load(loaded_results, load))


def check_H_load_results():
    """
    Checks the load v throughput results for the H topology
    :return: None
    """
    result_dir = 'published_results/H_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    results = load_results_from_files(files)
    print("Constructing H load simulation results from {} datapoints".format(len(results.keys())))
    plot_load_v_throughput_results(results)


def check_H_throughput_cdf():
    """
    Checks the throughput CDFs for the H graph simulations
    :return: None
    """
    # files = ["remote_results/H_load_results{}.json".format(i) for i in range(21) if i != 18]
    result_dir = 'published_results/H_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    loaded_results = load_results_from_files(files)
    for load in [str(i) for i in range(5, 55)]:
        import pdb
        pdb.set_trace()
        results = get_entries_for_load(loaded_results, load)
        print("Constructing H simulation plots from {} datapoints".format(len(results.keys())))
        plot_achieved_throughput(results)


def check_H_throughput_jitter():
    """
    Plots the achieved throughput/jitter pairs based on load for the H graph simulations
    :return: None
    """
    # files = ["remote_results/H_load_results{}.json".format(i) for i in range(21) if i != 18]
    result_dir = 'published_results/H_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    loaded_results = load_results_from_files(files)
    for load in [str(i) for i in range(5, 55)]:
        import pdb
        pdb.set_trace()
        results = get_entries_for_load(loaded_results, load)
        print("Constructing H simulation plots from {} datapoints".format(len(results.keys())))
        plot_throughput_jitter_hist2d(results)


def check_line_load_results():
    """
    Checks the load v throughput results for the line topology
    :return: None
    """
    result_dir = 'published_results/line_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    results = load_results_from_files(files)
    print("Constructing line load simulation results from {} datapoints".format(len(results.keys())))
    plot_load_v_throughput_results(results)


def check_line_throughput_cdf():
    """
    Checks the throughput CDFs based on load for the line graph simulations
    :return: None
    """
    # files = ["remote_results/line_load_results{}.json".format(i) for i in [0, 1, 2, 3, 4, 5]]
    result_dir = 'published_results/line_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    loaded_results = load_results_from_files(files)
    for load in [str(i) for i in range(5, 55)]:
        import pdb
        pdb.set_trace()
        results = get_entries_for_load(loaded_results, load)
        print("Constructing line simulation plots from {} datapoints".format(len(results.keys())))
        plot_achieved_throughput(results)


def check_line_throughput_jitter():
    """
    Plots the achieved throughput/jitter pairs based on load for the line graph simulations
    :return: None
    """
    # files = ["remote_results/line_load_results{}.json".format(i) for i in [0, 1, 2, 3, 4, 5]]
    result_dir = 'published_results/line_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    loaded_results = load_results_from_files(files)
    for load in [str(i) for i in range(5, 55)]:
        import pdb
        pdb.set_trace()
        results = get_entries_for_load(loaded_results, load)
        print("Constructing line simulation plots from {} datapoints".format(len(results.keys())))
        plot_throughput_jitter_hist2d(results)


def check_surfnet_load_results():
    """
    Checks the load v throughput results for the H topology
    :return: None
    """
    # files = ["results/surfnet_results/surfnet_load_results{}.json".format(i) for i in range(0, 40)]
    result_dir = 'published_results/surfnet_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    results = load_results_from_files(files)
    print("Constructing surfnet load simulation results from {} datapoints".format(len(results.keys())))
    plot_load_v_throughput_results(results)


def check_surfnet_throughput_cdf():
    """
    Checks the throughput CDFs for the H graph simulations
    :return: None
    """
    # files = ["results/surfnet_results/surfnet_load_results{}.json".format(i) for i in range(0, 40)]
    result_dir = 'published_results/surfnet_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    loaded_results = load_results_from_files(files)
    for load in [str(i) for i in range(5, 55)]:
        import pdb
        pdb.set_trace()
        results = get_entries_for_load(loaded_results, load)
        print("Constructing surfnet simulation plots from {} datapoints".format(len(results.keys())))
        plot_achieved_throughput(results)


def check_surfnet_throughput_jitter():
    """
    Plots the achieved throughput/jitter pairs based on load for the H graph simulations
    :return: None
    """
    # files = ["results/surfnet_results/surfnet_load_results{}.json".format(i) for i in range(0, 40)]
    result_dir = 'published_results/surfnet_results/reformatted/'
    files = [result_dir + filename for filename in listdir(result_dir) if ".json" in filename]
    loaded_results = load_results_from_files(files)
    for load in [str(i) for i in range(5, 55)]:
        import pdb
        pdb.set_trace()
        results = get_entries_for_load(loaded_results, load)
        print("Constructing surfnet simulation plots from {} datapoints".format(len(results.keys())))
        plot_throughput_jitter_hist2d(results)
