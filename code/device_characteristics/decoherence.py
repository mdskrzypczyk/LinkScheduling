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


def create_link_between_nodes(node1, loc1, node2, loc2, track=True, track_timestep=10000):
    # Create bell pair
    q1, q2 = create_qubits(2)
    operate([q1], H)
    operate([q1, q2], CNOT)

    # Store into devices
    device1 = node1.qmemory
    device2 = node2.qmemory
    device1.execute_instruction(INSTR_INIT, [loc1], qubit=q1, physical=False)
    device2.execute_instruction(INSTR_INIT, [loc2], qubit=q2, physical=False)
    tracker = None
    if track:
        tracker = FidelityTrackingProtocol(time_step=track_timestep, node=node1, qubit_id=loc1)

    return tracker


def perform_swap(swap_node, trackers=None, track_node=None, track_loc=None, track_timestep=10000):
    device = swap_node.qmemory
    q1, q2 = device.peek([0, 1])
    operate([q1, q2], CNOT)
    operate([q1], H)
    measure(q1)
    measure(q2)

    if trackers:
        for tracker in trackers:
            tracker.stop()

    tracker = None
    if track_node:
        tracker = FidelityTrackingProtocol(time_step=track_timestep, node=track_node, qubit_id=track_loc)

    return tracker


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


class FidelityTrackingProtocol(TimedProtocol):
    def __init__(self, node, qubit_id, time_step):
        super(FidelityTrackingProtocol, self).__init__(time_step=time_step, node=node)
        self.qubit_id = qubit_id
        self.timestamps = []
        self.fidelities = []
        self.start()

    def _execute(self):
        # Function to be executed when time elapsed. Will reschedule. Do not subclass this
        # to form a timed protocol, but subclass run_protocol instead.
        if self._running:
            self.run_protocol()
            # Schedule next execution
            self._schedule_after(self.time_step, self._EVT_TIME_TRIGGER)

    def run_protocol(self):
        device = self.node.qmemory
        qubit = device.peek(self.qubit_id)[0]
        if qubit is not None:
            F = max([fidelity(qubits=qubit.qstate._qubits, reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
            t = simutil.sim_time()
            self.timestamps.append(t)
            self.fidelities.append(F)


def single_vs_double_device_decoherence_no_noise():
    config_path = "/home/mds/Documents/projects/LinkScheduling/code/device_characteristics/config/two_uniform_device.json"
    network = setup_physical_network(config_file=config_path)
    node0 = network.get_node_by_id(0)
    device0 = node0.components['alice']
    node1 = network.get_node_by_id(1)
    device1 = node1.components['bob']

    # Single device decoherence in electron and carbon
    q1, q2 = create_qubits(2)
    q3, q4 = create_qubits(2)
    operate([q1], H)
    operate([q1, q2], CNOT)

    operate([q3], H)
    operate([q3, q4], CNOT)

    device0.execute_instruction(INSTR_INIT, [0], qubit=q1, physical=False)
    device0.execute_instruction(INSTR_INIT, [1], qubit=q3, physical=False)
    simutil.sim_reset()

    total_time = 2*simutil.SECOND  # 2s
    num_points = 20
    timestep = total_time / num_points
    timesteps = []
    fidelities_single_device_electron = []
    fidelities_single_device_carbon = []
    for i in range(num_points):
        timesteps.append(i*timestep)
        # Compute fidelity
        F = max([fidelity(qubits=[q1, q2], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        fidelities_single_device_electron.append(F)

        # Compute fidelity
        F = max([fidelity(qubits=[q3, q4], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        fidelities_single_device_carbon.append(F)
        simutil.sim_run(duration=timestep)
        device0.peek(0)
        device0.peek(1)

    # Double device decoherence in electrons and carbons
    q1, q2 = create_qubits(2)
    q3, q4 = create_qubits(2)
    operate([q1], H)
    operate([q1, q2], CNOT)

    operate([q3], H)
    operate([q3, q4], CNOT)

    device0.execute_instruction(INSTR_INIT, [0], qubit=q1, physical=False)
    device0.execute_instruction(INSTR_INIT, [1], qubit=q3, physical=False)
    device1.execute_instruction(INSTR_INIT, [0], qubit=q2, physical=False)
    device1.execute_instruction(INSTR_INIT, [1], qubit=q4, physical=False)

    fidelities_double_device_electron = []
    fidelities_double_device_carbon = []
    for i in range(num_points):
        # Compute fidelity
        F = max([fidelity(qubits=[q1, q2], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        fidelities_double_device_electron.append(F)

        # Compute fidelity
        F = max([fidelity(qubits=[q3, q4], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        fidelities_double_device_carbon.append(F)
        simutil.sim_run(duration=timestep)
        device0.peek(0)
        device1.peek(0)
        device0.peek(1)
        device1.peek(1)

    # Double device decoherence in electron and carbon
    q1, q2 = create_qubits(2)
    operate([q1], H)
    operate([q1, q2], CNOT)

    device0.execute_instruction(INSTR_INIT, [0], qubit=q1, physical=False)
    device1.execute_instruction(INSTR_INIT, [1], qubit=q2, physical=False)

    fidelities_double_device_electron_and_carbon = []
    for i in range(num_points):
        # Compute fidelity
        F = max([fidelity(qubits=[q1, q2], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        fidelities_double_device_electron_and_carbon.append(F)
        simutil.sim_run(duration=timestep)
        device0.peek(0)
        device1.peek(1)

    # Fit curves of decoherence
    def func(x, b):
        return 0.5 * np.exp(-b * x) + 0.5


    fit_timesteps = np.array(timesteps)
    fit_timesteps /= simutil.SECOND
    optimizedParameters_single_electron, pcov_single = opt.curve_fit(f=func, xdata=fit_timesteps,
                                                                     ydata=fidelities_single_device_electron,
                                                                     maxfev=10000)
    optimizedParameters_single_carbon, pcov_single = opt.curve_fit(f=func, xdata=fit_timesteps,
                                                                   ydata=fidelities_single_device_carbon,
                                                                   maxfev=10000)
    optimizedParameters_double_electron, pcov_double = opt.curve_fit(f=func, xdata=fit_timesteps,
                                                                     ydata=fidelities_double_device_electron,
                                                                     maxfev=10000)
    optimizedParameters_double_carbon, pcov_double = opt.curve_fit(f=func, xdata=fit_timesteps,
                                                                   ydata=fidelities_double_device_carbon,
                                                                   maxfev=10000)
    optimizedParameters_double_electron_and_carbon, pcov_double = opt.curve_fit(f=func, xdata=fit_timesteps,
                                                                                ydata=fidelities_double_device_electron_and_carbon,
                                                                                maxfev=10000)

    # Convert to microseconds
    plt.plot(timesteps, fidelities_single_device_electron, 'bo', label="true_single_electron")
    plt.plot(timesteps, func(fit_timesteps, *optimizedParameters_single_electron), color='b', label="fit_single_electron")
    plt.plot(timesteps, fidelities_double_device_electron, 'ro', label="true_double_electron")
    plt.plot(timesteps, func(fit_timesteps, *optimizedParameters_double_electron), color='r', label="fit_double_electron")
    plt.plot(timesteps, fidelities_single_device_carbon, 'go', label="true_single_carbon")
    plt.plot(timesteps, func(fit_timesteps, *optimizedParameters_single_carbon), color='g', label="fit_single_carbon")
    plt.plot(timesteps, fidelities_double_device_carbon, 'co', label="true_double_carbon")
    plt.plot(timesteps, func(fit_timesteps, *optimizedParameters_double_carbon), color='c', label="fit_double_carbon")
    plt.plot(timesteps, fidelities_double_device_electron_and_carbon, 'yo', label="true_double_electron_and_carbon")
    plt.plot(timesteps, func(fit_timesteps, *optimizedParameters_double_electron_and_carbon), color='y', label="fit_double_electron_and_carbon")
    plt.legend()
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Fidelity")
    plt.show()
    pdb.set_trace()


def _build_move_program():
    """Build and return a move program"""
    prgm = QuantumProgram()
    qs = prgm.get_qubit_indices(2)
    qprgms.move_using_CXDirections(prgm, qs[0], qs[1])
    return prgm


def delayed_move_decoherence():
    config_path = "/Users/mskrzypczyk/Documents/projects/LinkScheduling/code/linkscheduling/config/five_uniform_device.json"
    network = setup_physical_network(config_file=config_path)
    node0 = network.get_node_by_id(0)
    device0 = node0.components['alice']
    node1 = network.get_node_by_id(1)
    device1 = node1.components['bob']
    node2 = network.get_node_by_id(2)
    device2 = node2.components['charlie']
    node3 = network.get_node_by_id(3)
    device3 = node3.components['david']
    node4 = network.get_node_by_id(4)
    device4 = node4.components['eve']

    simutil.sim_reset()
    prgm = _build_move_program()

    total_time = 2*simutil.SECOND  # 3s
    move_delay = 2000*simutil.MILLISECOND
    num_points = 20
    timestep = total_time / num_points

    fmsp = []
    fmse = []
    fmde = []

    q1, q2 = create_qubits(2)
    q3, q4 = create_qubits(2)
    q5, q6 = create_qubits(2)

    operate([q1], H)
    operate([q1, q2], CNOT)

    operate([q3], H)
    operate([q3, q4], CNOT)

    operate([q5], H)
    operate([q5, q6], CNOT)

    a0, a1, a2, a3, a4 = create_qubits(5)

    device0.execute_instruction(INSTR_INIT, [0], qubit=q1, physical=False)
    device0.execute_instruction(INSTR_INIT, [1], qubit=a0, physical=False)

    device1.execute_instruction(INSTR_INIT, [0], qubit=q3, physical=False)
    device1.execute_instruction(INSTR_INIT, [1], qubit=a1, physical=False)

    device2.execute_instruction(INSTR_INIT, [0], qubit=q4, physical=False)
    device2.execute_instruction(INSTR_INIT, [1], qubit=a2, physical=False)

    device3.execute_instruction(INSTR_INIT, [0], qubit=q5, physical=False)
    device3.execute_instruction(INSTR_INIT, [1], qubit=a3, physical=False)

    device4.execute_instruction(INSTR_INIT, [0], qubit=q6, physical=False)
    device4.execute_instruction(INSTR_INIT, [1], qubit=a4, physical=False)

    simutil.sim_run(duration=move_delay)
    device0.peek(0)
    device0.peek(1)
    device1.peek(0)
    device1.peek(1)
    device2.peek(0)
    device2.peek(1)
    device3.peek(0)
    device3.peek(1)
    device4.peek(0)
    device4.peek(1)

    device0_noise = [mp.noise_model for mp in device0._memory_positions[0:2]]
    device0._memory_positions[0].noise_model = None
    device0._memory_positions[1].noise_model = None
    device0.execute_program(prgm, qubit_mapping=[0, 1])

    device1_noise = [mp.noise_model for mp in device1._memory_positions[0:2]]
    device1._memory_positions[0].noise_model = None
    device1._memory_positions[1].noise_model = None
    device1.execute_program(prgm, qubit_mapping=[0, 1])

    device3_noise = [mp.noise_model for mp in device3._memory_positions[0:2]]
    device3._memory_positions[0].noise_model = None
    device3._memory_positions[1].noise_model = None
    device3.execute_program(prgm, qubit_mapping=[0, 1])

    device4_noise = [mp.noise_model for mp in device4._memory_positions[0:2]]
    device4._memory_positions[0].noise_model = None
    device4._memory_positions[1].noise_model = None
    device4.execute_program(prgm, qubit_mapping=[0, 1])

    run_to_move_completion(device0)

    device0._memory_positions[0].noise_model = device0_noise[0]
    device0._memory_positions[1].noise_model = device0_noise[1]
    device1._memory_positions[0].noise_model = device1_noise[0]
    device1._memory_positions[1].noise_model = device1_noise[1]
    device3._memory_positions[0].noise_model = device3_noise[0]
    device3._memory_positions[1].noise_model = device3_noise[1]
    device4._memory_positions[0].noise_model = device4_noise[0]
    device4._memory_positions[1].noise_model = device4_noise[1]

    timesteps = []
    for r in range(num_points):
        timesteps.append(r*timestep)
        device0.peek(0)
        device0.peek(1)
        device1.peek(0)
        device1.peek(1)
        device2.peek(0)
        device2.peek(1)
        device3.peek(0)
        device3.peek(1)
        device4.peek(0)
        device4.peek(1)
        F = max([fidelity(qubits=[a0, q2], reference_ket=b, squared=True) for b in [sb00, sb01, sb10, sb11]])
        fmsp.append(F)
        F = max([fidelity(qubits=[a1, q4], reference_ket=b, squared=True) for b in [sb00, sb01, sb10, sb11]])
        fmse.append(F)
        F = max([fidelity(qubits=[a3, a4], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        fmde.append(F)
        simutil.sim_run(duration=timestep)

    # Fit curves of decoherence
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    fit_timesteps = np.array(timesteps)
    fit_timesteps /= simutil.SECOND
    optimizedParameters_msp, pcov_single = opt.curve_fit(f=func, xdata=fit_timesteps, ydata=fmsp, maxfev=10000)
    optimizedParameters_mse, pcov_single = opt.curve_fit(f=func, xdata=fit_timesteps, ydata=fmse, maxfev=10000)
    optimizedParameters_mde, pcov_double = opt.curve_fit(f=func, xdata=fit_timesteps, ydata=fmde, maxfev=10000)

    # Plot
    timesteps = np.array(timesteps) / simutil.MILLISECOND
    plt.plot(timesteps, fmsp, 'bo', label="true_msp")
    plt.plot(timesteps, func(fit_timesteps, *optimizedParameters_msp), color='b', label="fit_msp")
    plt.plot(timesteps, fmse, 'ro', label="true_mse")
    plt.plot(timesteps, func(fit_timesteps, *optimizedParameters_mse), color='r', label="fit_mse")
    plt.plot(timesteps, fmde, 'go', label="true_mde")
    plt.plot(timesteps, func(fit_timesteps, *optimizedParameters_mde), color='g', label="fit_mde")
    plt.legend()
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Fidelity")
    plt.show()
    pdb.set_trace()


def move_decoherence():
    config_path = "/Users/mskrzypczyk/Documents/projects/LinkScheduling/code/linkscheduling/config/five_uniform_device.json"
    network = setup_physical_network(config_file=config_path)
    node0 = network.get_node_by_id(0)
    device0 = node0.components['alice']
    node1 = network.get_node_by_id(1)
    device1 = node1.components['bob']
    node2 = network.get_node_by_id(2)
    device2 = node2.components['charlie']
    node3 = network.get_node_by_id(3)
    device3 = node3.components['david']
    node4 = network.get_node_by_id(4)
    device4 = node4.components['eve']


    simutil.sim_reset()
    prgm = _build_move_program()

    total_time = 3000000000  # 3s
    num_points = 20
    timestep = total_time / num_points

    ifmsp = []
    pfmsp = []
    ifmse = []
    pfmse = []
    ifmde = []
    pfmde = []
    for r in range(num_points):
        q1, q2 = create_qubits(2)
        q3, q4 = create_qubits(2)
        q5, q6 = create_qubits(2)

        operate([q1], H)
        operate([q1, q2], CNOT)

        operate([q3], H)
        operate([q3, q4], CNOT)

        operate([q5], H)
        operate([q5, q6], CNOT)

        a0, a1, a2, a3, a4 = create_qubits(5)

        device0.execute_instruction(INSTR_INIT, [0], qubit=q1, physical=False)
        device0.execute_instruction(INSTR_INIT, [1], qubit=a0, physical=False)

        device1.execute_instruction(INSTR_INIT, [0], qubit=q3, physical=False)
        device1.execute_instruction(INSTR_INIT, [1], qubit=a1, physical=False)

        device2.execute_instruction(INSTR_INIT, [0], qubit=q4, physical=False)
        device2.execute_instruction(INSTR_INIT, [1], qubit=a2, physical=False)

        device3.execute_instruction(INSTR_INIT, [0], qubit=q5, physical=False)
        device3.execute_instruction(INSTR_INIT, [1], qubit=a3, physical=False)

        device4.execute_instruction(INSTR_INIT, [0], qubit=q6, physical=False)
        device4.execute_instruction(INSTR_INIT, [1], qubit=a4, physical=False)

        simutil.sim_run(duration=r*timestep)
        device0.peek(0)
        device0.peek(1)
        device1.peek(0)
        device1.peek(1)
        device2.peek(0)
        device2.peek(1)
        device3.peek(0)
        device3.peek(1)
        device4.peek(0)
        device4.peek(1)
        F = max([fidelity(qubits=[q1, q2], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        ifmsp.append(F)
        F = max([fidelity(qubits=[q3, q4], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        ifmse.append(F)
        F = max([fidelity(qubits=[q5, q6], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        ifmde.append(F)

        device0_noise = [mp.noise_model for mp in device0._memory_positions[0:2]]
        device0._memory_positions[0].noise_model = None
        device0._memory_positions[1].noise_model = None
        device0.execute_program(prgm, qubit_mapping=[0, 1])

        device1_noise = [mp.noise_model for mp in device1._memory_positions[0:2]]
        device1._memory_positions[0].noise_model = None
        device1._memory_positions[1].noise_model = None
        device1.execute_program(prgm, qubit_mapping=[0, 1])

        device3_noise = [mp.noise_model for mp in device3._memory_positions[0:2]]
        device3._memory_positions[0].noise_model = None
        device3._memory_positions[1].noise_model = None
        device3.execute_program(prgm, qubit_mapping=[0, 1])

        device4_noise = [mp.noise_model for mp in device4._memory_positions[0:2]]
        device4._memory_positions[0].noise_model = None
        device4._memory_positions[1].noise_model = None
        device4.execute_program(prgm, qubit_mapping=[0, 1])

        run_to_move_completion(device0)
        device0.peek(0)
        device0.peek(1)
        device1.peek(0)
        device1.peek(1)
        device2.peek(0)
        device2.peek(1)
        device3.peek(0)
        device3.peek(1)
        device4.peek(0)
        device4.peek(1)

        F = max([fidelity(qubits=[a0, q2], reference_ket=b, squared=True) for b in [sb00, sb01, sb10, sb11]])
        pfmsp.append(F)
        F = max([fidelity(qubits=[a1, q4], reference_ket=b, squared=True) for b in [sb00, sb01, sb10, sb11]])
        pfmse.append(F)
        F = max([fidelity(qubits=[a3, a4], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        pfmde.append(F)

        device0._memory_positions[0].noise_model = device0_noise[0]
        device0._memory_positions[1].noise_model = device0_noise[1]
        device1._memory_positions[0].noise_model = device1_noise[0]
        device1._memory_positions[1].noise_model = device1_noise[1]
        device3._memory_positions[0].noise_model = device3_noise[0]
        device3._memory_positions[1].noise_model = device3_noise[1]
        device4._memory_positions[0].noise_model = device4_noise[0]
        device4._memory_positions[1].noise_model = device4_noise[1]

    # Fit curves of decoherence
    def func(x, a, b):
        return a * x + b

    optimizedParameters_msp, pcov_single = opt.curve_fit(f=func, xdata=np.array(ifmsp), ydata=pfmsp)
    optimizedParameters_mse, pcov_single = opt.curve_fit(f=func, xdata=np.array(ifmse), ydata=pfmse)
    optimizedParameters_mde, pcov_double = opt.curve_fit(f=func, xdata=np.array(ifmde), ydata=pfmde)

    # Convert to microseconds
    plt.plot(ifmsp, pfmsp, 'bo', label="move_single_perfect")
    plt.plot(ifmsp, func(np.array(ifmsp), *optimizedParameters_msp), color='b', label="fit_single_electron")
    plt.plot(ifmse, pfmse, 'ro', label="move_single_electron")
    plt.plot(ifmse, func(np.array(ifmse), *optimizedParameters_mse), color='r', label="fit_double_electron")
    plt.plot(ifmde, pfmde, 'go', label="move_double")
    plt.plot(ifmde, func(np.array(ifmde), *optimizedParameters_mde), color='g', label="fit_single_electron")
    plt.legend()
    plt.xlabel("Input Fidelity")
    plt.ylabel("Output Fidelity")
    plt.show()
    pdb.set_trace()


def swap_decoherence():
    config_path = "/Users/mskrzypczyk/Documents/projects/LinkScheduling/code/linkscheduling/config/five_uniform_device.json"
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
    plt.plot(timesteps, ifl, 'bo', label="Fidelity electron-electron")
    plt.plot(timesteps, ifr, 'ro', label="Fidelity carbon-electron")
    plt.plot(timesteps, sf, 'go', label="Swapped Fidelity")
    plt.plot(timesteps, func((np.array(ifl), np.array(ifr)), *optimizedParameters_sf), 'g', label="Fit Swapped Fidelity")
    plt.plot(timesteps, ideal_decoherence(np.array(ifl), np.array(ifr)), 'y', label="Ideal Swapped Fidelity")
    plt.legend()
    plt.xlabel("Swap Delay (ms)")
    plt.ylabel("Fidelity")
    plt.show()
    pdb.set_trace()


if __name__ == '__main__':
    single_vs_double_device_decoherence_no_noise()
