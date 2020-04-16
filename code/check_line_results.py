from os import listdir
files = ["line_results/{}".format(file) for file in listdir("line_results")]
from simulations.analysis import load_results_from_files
from simulations.analysis import plot_results
results = load_results_from_files(files)
plot_results(results)
