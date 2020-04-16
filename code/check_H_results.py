from os import listdir
files = ["H_results/{}".format(file) for file in listdir("H_results")]
from simulations.analysis import load_results_from_files
from simulations.analysis import plot_results
results = load_results_from_files(files)
plot_results(results)
