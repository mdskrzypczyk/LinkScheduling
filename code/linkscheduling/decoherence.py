import pdb
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
from netsquid import simutil
from netsquid.components.instructions import INSTR_INIT
from netsquid.protocols import TimedProtocol
from netsquid.qubits.qubitapi import create_qubits, measure, operate, set_qstate_formalism, QFormalism, fidelity
from netsquid.qubits.ketstates import b00, b01, b10, b11
from netsquid.qubits.operators import CNOT, H, X, Z
from netsquid_netconf.easynetwork import setup_physical_network

set_qstate_formalism(QFormalism.DM)


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


def single_vs_double_device_decoherence():
    config_path = "/Users/mskrzypczyk/Documents/projects/LinkScheduling/code/linkscheduling/config/two_uniform_device.json"
    network = setup_physical_network(config_file=config_path)
    node0 = network.get_node_by_id(0)
    device0 = node0.components['alice']
    node1 = network.get_node_by_id(1)
    device1 = node1.components['bob']

    # Single device decoherence
    q1, q2 = create_qubits(2)
    operate([q1], H)
    operate([q1, q2], CNOT)

    device0.execute_instruction(INSTR_INIT, [0], qubit=q1, physical=False)
    simutil.sim_reset()

    total_time = 2000000  # 2ms
    num_points = 20
    timesteps = np.linspace(0, total_time, num_points)
    fidelities_single_device = []
    for timestep in timesteps:
        simutil.sim_run(end_time=timestep)
        dm = device0.peek(0)[0].qstate.dm
        # Compute fidelity
        F = max([fidelity(qubits=[q1, q2], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        fidelities_single_device.append(F)


    device0.pop(0)
    # Double device decoherence
    q1, q2 = create_qubits(2)
    operate([q1], H)
    operate([q1, q2], CNOT)

    device0.execute_instruction(INSTR_INIT, [0], qubit=q1, physical=False)
    device1.execute_instruction(INSTR_INIT, [0], qubit=q2, physical=False)

    fidelities_double_device = []
    for timestep in timesteps:
        simutil.sim_run(end_time=total_time + timestep)
        dm = device0.peek(0)[0].qstate.dm
        device1.peek(0)
        # Compute fidelity
        F = max([fidelity(qubits=[q1, q2], reference_ket=b, squared=True) for b in [b00, b01, b10, b11]])
        fidelities_double_device.append(F)


    # Fit curves of decoherence
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c


    fit_timesteps = timesteps / (total_time / num_points)
    optimizedParameters_single, pcov_single = opt.curve_fit(f=func, xdata=fit_timesteps, ydata=fidelities_single_device)
    optimizedParameters_double, pcov_double = opt.curve_fit(f=func, xdata=fit_timesteps, ydata=fidelities_double_device)

    # Convert to microseconds
    plot_timesteps = timesteps / 1000
    plt.plot(plot_timesteps, fidelities_single_device, 'bo', label="true_single")
    plt.plot(plot_timesteps, func(fit_timesteps, *optimizedParameters_single), label="fit_single")
    plt.plot(plot_timesteps, fidelities_double_device, 'ro', label="true_double")
    plt.plot(plot_timesteps, func(fit_timesteps, *optimizedParameters_double), label="fit_double")
    plt.legend()
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Fidelity")
    plt.show()
    pdb.set_trace()


def three_device_uniform_decoherence():
    config_path = "/Users/mskrzypczyk/Documents/projects/LinkScheduling/code/linkscheduling/config/three_uniform_device.json"

    # Set up the network
    simutil.sim_reset()
    network = setup_physical_network(config_file=config_path)
    node0 = network.get_node_by_id(0)
    device0 = node0.qmemory
    node1 = network.get_node_by_id(1)
    device1 = node1.qmemory
    node2 = network.get_node_by_id(2)
    device2 = node2.qmemory

    # Add Protocols to track fidelity
    time_step = 10000
    p0 = FidelityTrackingProtocol(time_step=time_step, node=node0, qubit_id=0)

    # First "create" the first bell pair and keep bob's in a carbon
    q1, q2 = create_bell_pair()
    device0.execute_instruction(INSTR_INIT, [0], qubit=q1, physical=False)
    device1.execute_instruction(INSTR_INIT, [1], qubit=q2, physical=False)

    # Time to generate second link and delay before executing swap
    link_gen_time = 200000

    # Simulate decoherence of first link
    simutil.sim_run(duration=link_gen_time)

    # Next "create" the second bell pair and store this one in bob's electron
    q3, q4 = create_bell_pair()
    device1.execute_instruction(INSTR_INIT, [0], qubit=q3, physical=False)
    device2.execute_instruction(INSTR_INIT, [0], qubit=q4, physical=False)
    p1 = FidelityTrackingProtocol(time_step=time_step, node=node1, qubit_id=0)

    # Time before the node executes the SWAP locally
    swap_delay = 30000

    # Simulate decoherence of both links before SWAP is executed
    simutil.sim_run(duration=swap_delay)

    # Perform the swap
    q1, q2 = device1.peek([0, 1])
    qc = device0.peek(0)[0]
    perform_swap(q1, q2, qc)

    p0.stop()
    p2 = FidelityTrackingProtocol(time_step=time_step, node=node0, qubit_id=0)

    # Track how link evolves post-SWAP
    evolution_time = 200000
    simutil.sim_run(duration=evolution_time)

    # Convert to microseconds
    plt.plot(p0.timestamps, p0.fidelities, 'bo')
    plt.plot(p1.timestamps, p1.fidelities, 'ro')
    plt.plot(p2.timestamps, p2.fidelities, 'go')
    plt.legend()
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Fidelity")
    plt.show()
    pdb.set_trace()


def four_device_uniform_decoherence():
    config_path = "/Users/mskrzypczyk/Documents/projects/LinkScheduling/code/linkscheduling/config/four_uniform_device.json"

    # Set up the network
    simutil.sim_reset()
    network = setup_physical_network(config_file=config_path)
    node0 = network.get_node_by_id(0)
    device0 = node0.qmemory
    node1 = network.get_node_by_id(1)
    device1 = node1.qmemory
    node2 = network.get_node_by_id(2)
    device2 = node2.qmemory
    node3 = network.get_node_by_id(3)
    device3 = node3.qmemory

    protocol_time_step = 10000

    # First "create" the first bell pair and keep bob's in a carbon
    q1, q2 = create_bell_pair()
    device0.execute_instruction(INSTR_INIT, [0], qubit=q1, physical=False)
    device1.execute_instruction(INSTR_INIT, [1], qubit=q2, physical=False)
    p0 = FidelityTrackingProtocol(time_step=protocol_time_step, node=node0, qubit_id=0)

    # Delay the generation of the second bell pair and simulate decoherence
    link_gen_time = 50000
    simutil.sim_run(duration=link_gen_time)

    # "Create" the second pair between the last two nodes
    q2, q3 = create_bell_pair()
    device2.execute_instruction(INSTR_INIT, [1], qubit=q2, physical=False)
    device3.execute_instruction(INSTR_INIT, [0], qubit=q3, physical=False)
    p1 = FidelityTrackingProtocol(time_step=protocol_time_step, node=node3, qubit_id=0)

    # Now nodes 1 and 2 can generate a link with one another, simulate decoherence
    link_gen_time = 200000
    simutil.sim_run(duration=link_gen_time)

    # Next "create" the second bell pair and store this one in bob's electron
    q5, q6 = create_bell_pair()
    device1.execute_instruction(INSTR_INIT, [0], qubit=q5, physical=False)
    device2.execute_instruction(INSTR_INIT, [0], qubit=q6, physical=False)
    p2 = FidelityTrackingProtocol(time_step=protocol_time_step, node=node1, qubit_id=0)

    # Time before the node executes the SWAP locally
    swap_delay = 30000

    # Simulate decoherence of both links before SWAP is executed
    simutil.sim_run(duration=swap_delay)

    # Perform the teleportation
    q1, q2 = device1.peek([0, 1])
    qc = device0.peek(0)[0]
    perform_swap(q1, q2, qc)
    q1, q2 = device2.peek([0, 1])
    qc = device3.peek(0)[0]
    perform_swap(q1, q2, qc)

    # Stop tracking swapped links
    p0.stop()
    p1.stop()
    p2.stop()
    p3 = FidelityTrackingProtocol(time_step=protocol_time_step, node=node0, qubit_id=0)

    # Track how link evolves post-SWAP
    evolution_time = 200000
    simutil.sim_run(duration=evolution_time)

    # Convert to microseconds
    plt.plot(p0.timestamps, p0.fidelities, 'bo', label="LinkAB")
    plt.plot(p1.timestamps, p1.fidelities, 'ro', label="LinkCD")
    plt.plot(p2.timestamps, p2.fidelities, 'go', label="LinkBC")
    plt.plot(p3.timestamps, p3.fidelities, 'co', label="LinkAD")
    plt.legend()
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Fidelity")
    plt.show()
    pdb.set_trace()


def five_device_uniform_decoherence():
    config_path = "/Users/mskrzypczyk/Documents/projects/LinkScheduling/code/linkscheduling/config/five_uniform_device.json"

    # Set up the network
    simutil.sim_reset()
    network = setup_physical_network(config_file=config_path)
    node0 = network.get_node_by_id(0)
    node1 = network.get_node_by_id(1)
    node2 = network.get_node_by_id(2)
    node3 = network.get_node_by_id(3)
    node4 = network.get_node_by_id(4)

    # First "create" the first bell pair and keep bob's in a carbon
    p0 = create_link_between_nodes(node0, 0, node1, 1)

    # Delay the generation of the second bell pair and simulate decoherence
    link_gen_time = 50000
    simutil.sim_run(duration=link_gen_time)

    # "Create" the second pair between the last two nodes
    p1 = create_link_between_nodes(node2, 1, node3, 1)

    # Now nodes 1 and 2 can generate a link with one another, simulate decoherence
    link_gen_time = 20000
    simutil.sim_run(duration=link_gen_time)

    # Next "create" the second bell pair and store this one in bob's electron
    p2 = create_link_between_nodes(node1, 0, node2, 0)

    # Time before the node executes the SWAP locally
    swap_delay = 30000

    # Simulate decoherence of both links before SWAP is executed
    simutil.sim_run(duration=swap_delay)

    # SWAP the link
    p4 = perform_swap(node1, [p0, p2], node0, 0)
    perform_swap(node2)

    # Next "create" the second bell pair and store this one in bob's electron
    link_gen_time = 40000
    simutil.sim_run(duration=link_gen_time)
    p3 = create_link_between_nodes(node3, 0, node4, 0)

    simutil.sim_run(duration=swap_delay)
    perform_swap(node3)

    # Track how link evolves post-SWAP
    evolution_time = 200000
    simutil.sim_run(duration=evolution_time)

    # Convert to microseconds
    plt.plot(p0.timestamps, p0.fidelities, 'bo', label="LinkAB")
    plt.plot(p1.timestamps, p1.fidelities, 'ro', label="LinkCD")
    plt.plot(p2.timestamps, p2.fidelities, 'go', label="LinkBC")
    plt.plot(p3.timestamps, p3.fidelities, 'co', label="LinkDE")
    plt.plot(p4.timestamps, p4.fidelities, 'yo', label="LinkAE")
    plt.legend()
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Fidelity")
    plt.show()
    pdb.set_trace()


if __name__=='__main__':
    five_device_uniform_decoherence()


