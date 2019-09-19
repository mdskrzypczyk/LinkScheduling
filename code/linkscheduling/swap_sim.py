import pdb
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
from abc import ABCMeta
from netsquid import simutil
from netsquid.components.instructions import INSTR_INIT, INSTR_H, INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z, INSTR_CXDIR, INSTR_MEASURE
from netsquid.protocols import TimedProtocol
from netsquid.qubits.qubitapi import create_qubits, measure, operate, set_qstate_formalism, QFormalism, fidelity, reduced_dm
from netsquid.qubits.ketstates import b00, b01, b10, b11, s0, s1, h0, h1
from netsquid.qubits.operators import CNOT, H, X, Z
from netsquid_netconf.easynetwork import setup_physical_network
from netsquid.components.qprogram import QuantumProgram
from netsquid_physlayer import qProgramLibrary as qprgms


set_qstate_formalism(QFormalism.DM)

#custom kets
sh00 = np.kron(s0, h0)
sh01 = np.kron(s0, h1)
sh10 = np.kron(s1, h0)
sh11 = np.kron(s1, h1)

sb00 = (sh00 + sh11) / np.sqrt(2)
sb01 = (sh01 + sh10) / np.sqrt(2)
sb10 = (sh00 - sh11) / np.sqrt(2)
sb11 = (sh01 - sh10) / np.sqrt(2)


def run_to_move_completion(device):
    amount_of_time = 0
    while device.busy:
        amount_of_time += 1000
        simutil.sim_run(duration=1000)

    return amount_of_time


def reverse_move_using_CXDirections(q_program, control=0, target=1):
    """
    The reverse of the circuit defined in
    :obj:`~easysquid.qProgramLibrary.move_using_CXDirections`.
    """
    q_program.apply(INSTR_ROT_Y, control, angle=np.pi / 2)
    q_program.apply(INSTR_CXDIR, [control, target], angle=-np.pi / 2)
    q_program.apply(INSTR_ROT_X, control, angle=-np.pi / 2)
    # TODO I should still compute why the ROT_Z gates here
    # rotate over +pi/2 before the CXDIR and -pi/2 after
    # instead of the other way around
    q_program.apply(INSTR_ROT_Z, target, angle=np.pi / 2)
    q_program.apply(INSTR_CXDIR, [control, target], angle=np.pi / 2)
    q_program.apply(INSTR_ROT_Z, target, angle=-np.pi / 2)


class IBellStateMeasurementProgram(QuantumProgram, metaclass=ABCMeta):
    """
    Attributes
    -----------
    outcome : int
           The outcome of the Bell-state-measurement using
           the following translaten: 0=Phi^+, 1=Psi^+, 2=Phi^-, 3=Psi^-.

    Internal working
    ----------------
    A few private attributes:
      * _NAME_OUTCOME_CONTROL : str
      * _NAME_OUTCOME_TARGET : str
      * OUTCOME_TO_BELL_INDEX : dict with keys (int, int) and values int
           Indicates how the two measurement outcomes are related to the
           state that is measured. Its keys are tuples of the two measurement
           outcomes (control, target) and its values is the Bell state index
           using the following translaten: 0=Phi^+, 1=Psi^+, 2=Phi^-, 3=Psi^-.
    """

    default_num_qubits = 2
    _NAME_OUTCOME_CONTROL = "control-qubit-outcome"
    _NAME_OUTCOME_TARGET = "target-qubit-outcome"
    _OUTCOME_TO_BELL_INDEX = {(x, y): None for x in [0, 1] for y in [0, 1]}

    @property
    def get_outcome_as_bell_index(self):
        m_outcome_control = self.output[self._NAME_OUTCOME_CONTROL][0]
        m_outcome_target = self.output[self._NAME_OUTCOME_TARGET][0]
        return self._OUTCOME_TO_BELL_INDEX[(m_outcome_control, m_outcome_target)]


class RestrictedBSMProgram(IBellStateMeasurementProgram):
    """ Entanglement distillation (2017), Kalb et al.
    Reverse of fig. 2a"""

    _OUTCOME_TO_BELL_INDEX = {(0, 0): 2, (0, 1): 1, (1, 0): 0, (1, 1): 3}

    def program(self):
        electron, carbon = self.get_qubit_indices(2)
        self.apply(INSTR_ROT_Z, carbon, angle=np.pi/2)
        self.apply(INSTR_CXDIR, [electron, carbon], angle=np.pi/2)
        self.apply(INSTR_ROT_Z, carbon, angle=-np.pi/2)
        self.apply(INSTR_ROT_Y, electron, angle=np.pi/2)
        self.apply(INSTR_MEASURE, electron, self._NAME_OUTCOME_CONTROL, inplace=False)
        self.apply(INSTR_INIT, electron)
        # map the carbon state onto the electron
        reverse_move_using_CXDirections(q_program=self, control=electron, target=carbon)
        # We apply an additional hadamard to compensate for the fact that the
        # carbon state lives in a rotates basis (the Hadamard basis instead
        # of the computational basis)
        self.apply(INSTR_H, electron)
        self.apply(INSTR_MEASURE, electron, self._NAME_OUTCOME_TARGET, inplace=False)
        yield self.run()


def swap_decoherence():
    config_path = "config/swap_config.json"
    network = setup_physical_network(config_file=config_path)
    node0 = network.get_node_by_id(0)
    device0 = node0.components['alice']
    node1 = network.get_node_by_id(1)
    device1 = node1.components['bob']
    node2 = network.get_node_by_id(2)
    device2 = node2.components['charlie']

    simutil.sim_reset()
    prgm = RestrictedBSMProgram()

    total_time = 3000000000  # 3s
    num_points = 20
    timestep = total_time / num_points

    ifl = []
    ifr = []
    sf = []
    timesteps = []
    for r in range(num_points):
        q1, q2 = create_qubits(2)
        q3, q4 = create_qubits(2)

        operate([q1], H)
        operate([q1, q2], CNOT)

        operate([q3], H)
        operate([q3, q4], CNOT)

        a0, a1 = create_qubits(2)

        device0.execute_instruction(INSTR_INIT, [0], qubit=q1, physical=False)
        device0.execute_instruction(INSTR_INIT, [1], qubit=a0, physical=False)

        device1.execute_instruction(INSTR_INIT, [0], qubit=q2, physical=False)
        device1.execute_instruction(INSTR_INIT, [1], qubit=q3, physical=False)

        device2.execute_instruction(INSTR_INIT, [0], qubit=q4, physical=False)
        device2.execute_instruction(INSTR_INIT, [1], qubit=a1, physical=False)

        simutil.sim_run(duration=r * timestep)
        timesteps.append(r*timestep)
        device0.peek(0)
        device0.peek(1)
        device1.peek(0)
        device1.peek(1)
        device2.peek(0)
        device2.peek(1)
        F = max([fidelity(qubits=[q1, q2], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        ifl.append(F)
        F = max([fidelity(qubits=[q3, q4], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        ifr.append(F)

        device1_noise = [mp.noise_model for mp in device1._memory_positions[0:2]]
        device1._memory_positions[0].noise_model = None
        device1._memory_positions[1].noise_model = None
        device1.execute_program(prgm, qubit_mapping=[0, 1])
        run_to_move_completion(device1)

        device0.peek(0)
        device0.peek(1)
        device1.peek(0)
        device1.peek(1)
        device2.peek(0)
        device2.peek(1)

        F = max([fidelity(qubits=[q1, q4], reference_ket=b, squared=True) for b in [sb00, sb01, sb10, sb11]])
        sf.append(F)

        device1._memory_positions[0].noise_model = device1_noise[0]
        device1._memory_positions[1].noise_model = device1_noise[1]

    def ideal_decoherence(x, y):
        return x*y + (1-x)*(1-y)/3

    def func(X, a, b, c):
        x, y = X
        return a*x*y + b*(1-x)*(1-y)/3 + c

    optimizedParameters_sf, pcov_double = opt.curve_fit(f=func, xdata=(np.array(ifl), np.array(ifr)), ydata=sf)

    # Convert to microseconds
    timesteps = np.array(timesteps) / simutil.MILLISECOND
    plt.plot(timesteps, ifl, 'bo', label="Fidelity loc0-loc0")
    plt.plot(timesteps, ifr, 'ro', label="Fidelity loc1-loc0")
    plt.plot(timesteps, sf, 'go', label="Swapped Fidelity")
    plt.plot(timesteps, func((np.array(ifl), np.array(ifr)), *optimizedParameters_sf), 'g', label="Fit Swapped Fidelity")
    plt.plot(timesteps, ideal_decoherence(np.array(ifl), np.array(ifr)), 'y', label="Ideal Swapped Fidelity")
    plt.legend()
    plt.xlabel("Swap Delay (ms)")
    plt.ylabel("Fidelity")
    plt.show()
    pdb.set_trace()


if __name__ == '__main__':
    swap_decoherence()
