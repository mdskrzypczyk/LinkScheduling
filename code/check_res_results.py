from os import listdir
files = ["resource_results/{}".format(file) for file in listdir("resource_results")]
from simulations.analysis import load_results_from_files
from simulations.analysis import plot_pb_results
results = load_results_from_files(files)

plot_pb_results(results)
