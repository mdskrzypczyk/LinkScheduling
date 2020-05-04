"""
In this example, we simulate the benchmarking experiment from
"Deterministic delivery of remote entanglement" using
magically generated entanglement.
"""

import json
from netsquid.qubits.qubitapi import dm_fidelity
from netsquid.qubits.ketstates import b00, b01, b10, b11, ket2dm
from netsquid_nv.state_delivery_sampler_factory import NVStateDeliverySamplerFactory
from netsquid_nv.delft_nvs.delft_nv_2019_optimistic import NVParameterSet2019Optimistic
import numpy as np
import matplotlib.pyplot as plt


def main(no_output=False):
    """
    Simulation used for obtaining link capabilities of NV in Diamond
    :param no_output:
    :return:
    """
    gss_factory = NVStateDeliverySamplerFactory()

    d_to_rates = {}
    d_to_sem_rates = {}
    d_to_fidelities = {}
    d_to_sem_fidelities = {}
    alphas = list(np.linspace(0.1, 0.5, 10))
    ideal_states = [ket2dm(b) for b in [b00, b01, b10, b11]]
    distances = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for internode_distance in distances:
        rates = []  # mean rates
        fidelities = []
        sem_rates = []  # standard errors of the mean of the rate
        sem_fidelities = []

        # Perform sampling
        for alpha in alphas:
            current_params = NVParameterSet2019Optimistic()
            cycle_time = current_params.photon_emission_delay + \
                internode_distance / current_params.c * 10 ** 9
            gss = gss_factory.create_state_delivery_sampler(
                **current_params.to_dict(),
                p_det=current_params.to_dict()["prob_detect_excl_transmission_no_conversion_no_cavities"],
                cycle_time=cycle_time,
                alpha=alpha)

            number_of_samples = 1000

            # sampling
            data = [gss.sample() for __ in range(number_of_samples)]
            states = [sample[0] for sample in data]
            latencies = [sample[1] for sample in data]
            alpha_fidelities = [max([dm_fidelity(state, ideal, squared=True) for ideal in ideal_states]) for state in states]

            # Computing mean and standard deviation of the mean
            rates.append(1. / np.mean(latencies) * 10 ** 9)
            fidelities.append(np.mean(alpha_fidelities))
            std = np.std(latencies) * 10 ** (-9) / (np.mean(latencies) * 10 ** (-9)) ** 2
            sem_rates.append(std / np.sqrt(number_of_samples))
            std = np.std(fidelities) / (np.mean(fidelities)) ** 2
            sem_fidelities.append(std / np.sqrt(number_of_samples))

        print("Finished distance {}".format(internode_distance))
        d_to_rates[internode_distance] = rates
        d_to_sem_rates[internode_distance] = sem_rates
        d_to_fidelities[internode_distance] = fidelities
        d_to_sem_fidelities[internode_distance] = sem_fidelities

    data = {
        "rates": d_to_rates,
        "sem_rates": d_to_sem_rates,
        "fidelities": d_to_fidelities,
        "sem_fidelities": d_to_sem_fidelities
    }
    with open("data.json", "w") as f:
        json.dump(data, f)

    for internode_distance in distances:
        rates = d_to_rates[internode_distance]
        sem_rates = d_to_sem_rates[internode_distance]
        fidelities = d_to_fidelities[internode_distance]
        sem_fidelities = d_to_sem_fidelities[internode_distance]
        # Plotting
        if not no_output:
            plt.subplot(1, 2, 1)
            plt.title(
                "Simulating fig.2 from 'Deterministic delivery of remote entanglement'" +
                "\n(internode distance: {} m)".format(internode_distance*1e3))
            plt.errorbar(alphas, rates, yerr=sem_rates, capsize=4)
            plt.xlabel(r"Bright-state parameter ($\alpha$)")
            plt.ylabel("Entanglement generation rate (Hz)")

            plt.subplot(1, 2, 2)
            plt.title(
                "Simulating achievable fidelities" +
                "\n(internode distance: {} m)".format(internode_distance * 1e3))
            plt.errorbar(alphas, fidelities, yerr=sem_fidelities, capsize=4)
            plt.xlabel(r"Bright-state parameter ($\alpha$)")
            plt.ylabel("Fidelity")
            plt.show()

    return True


if __name__ == "__main__":
    main(no_output=False)
