from os import listdir
files = ["pb_results/{}".format(file) for file in listdir("pb_results")]
from simulations.analysis import load_results_from_files
from simulations.analysis import plot_pb_results
results = load_results_from_files(files)

plot_pb_results(results)
