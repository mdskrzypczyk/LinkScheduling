import json
import random
from math import ceil
from os.path import exists
from jobscheduling.log import LSLogger


logger = LSLogger()

SIMULATION_DEMAND_DATA = "simulation_demand_data.json"
SLOT_SIZE = 0.02  # 20ms

def get_simulation_topologies(end_node_resources, repeater_resources):
    pass

def generate_demands():
    end_node_resources = (1, 3)
    repeater_resource_configurations = [(1, 3), (1, 4), (2, 3), (2, 4)]

    # First generate the demands for the simulation
    for repeater_resources in repeater_resource_configurations:
        topologies = get_simulation_topologies(end_node_resources, repeater_resources)
        demand_data = load_demand_data()
        for topology in topologies:
            if demands_generated(demand_data, topology):
                continue
            topology_demands = generate_topology_demands(topology)
            demand_data = update_demand_data(demand_data, topology, topology_demands)

    return demand_data


def load_demand_data():
    if exists(SIMULATION_DEMAND_DATA):
        with open(SIMULATION_DEMAND_DATA) as f:
            demand_data = json.load(f)
    else:
        demand_data = {}

    return demand_data


def update_demand_data(demand_data, topology, topology_demands):
    topology_name = ...
    demand_data[topology_name] = topology_demands
    with open(SIMULATION_DEMAND_DATA, 'w') as f:
        json.dump(f, demand_data)
    return demand_data


def demands_generated(demand_data, topology):
    topology_name = ...
    if demand_data.get(topology_name) is not None:
        return True
    else:
        return False


def generate_topology_demands(topology):
    topology_demands = {}

    demand_fidelities = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # Get the end nodes
    end_node_resources = ...

    for demand_fidelity in demand_fidelities:
        fidelity_demands = []
        resource_utilizations = dict([(resource, 0.0) for resource in end_node_resources])

        # Record node resources in end nodes
        underutilized_resources = list(resource_utilizations.keys())
        while underutilized_resources:
            # Choose a node that has underutilized resources
            underutilized_resource = random.choice(underutilized_resources)
            source = underutilized_resource.split('-')[0]

            # Randomly choose another end node in the network
            destination = random.choice([node for node in topology.nodes if node != source])

            # Construct a concrete protocol between end nodes with demand fidelity
            protocol_task = get_concrete_protocol(topology, source, destination, demand_fidelity)
            if protocol_task is None:
                continue

            # Select a rate for the concrete protocol
            rate = choose_rate_for_demand(protocol_task)
            protocol_task.p = ceil(1 / rate / SLOT_SIZE)

            # Update resource utilizations
            protocol_resource_utilizations = get_protocol_resource_utilizations(protocol_task)
            for resource, utilization in protocol_resource_utilizations:
                resource_utilizations[resource] += utilization
                if resource_utilizations[resource] >= 1:
                    underutilized_resources.remove(resource)

            # Record the new demand
            fidelity_demands.append({"source": source, "destination": destination, "rate": rate})

        topology_demands[demand_fidelity] = fidelity_demands

    return topology_demands


def get_concrete_protocol(topology, source, destination, demand_fidelity):
    try:
        protocol = get_protocol_without_rate_constraint(topology, source, destination, demand_fidelity)
        if protocol is None:
            logger.warning("Demand {} could not be satisfied!".format((source, destination. fidelity)))
            return None

        task = convert_protocol_to_task(demand, protocol, SLOT_SIZE)

        scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, topology)

        latency = scheduled_task.c * SLOT_SIZE
        achieved_rate = 1 / latency

        new_rate = select_rate(achieved_rate)
        if new_rate == 0:
            logger.warning("Could not provide rate for {}".format(demand))
            continue

        scheduled_task.p = ceil(1 / new_rate / slot_size)

        s, d, f, r = demand
        demand = (s, d, f, new_rate)
        s, d, f, r = demand

        asap_dec, alap_dec, shift_dec = decoherence_times
        logger.info("Results for {}:".format(demand))
        if not correct:
            logger.error("Failed to construct valid protocol for {}".format(demand))
            import pdb
            pdb.set_trace()
        elif achieved_rate < r:
            logger.warning("Failed to satisfy rate for {}, achieved {}".format(demand, achieved_rate))
            import pdb
            pdb.set_trace()
        elif shift_dec > asap_dec or shift_dec > alap_dec:
            logger.error("Shifted protocol has greater decoherence than ALAP or ASAP for demand {}".format(demand))
            import pdb
            pdb.set_trace()
        else:
            logger.info("Successfully created protocol and task for demand (S={}, D={}, F={}, R={}), {}".format(*demand,
                                                                                                                num_succ))
            taskset.append(scheduled_task)

    except Exception as err:
        logger.exception("Error occurred while generating tasks: {}".format(err))


def get_protocol_without_rate_constraint(network_topology, s, d, f):
    _, nodeG = network_topology
    path = nx.shortest_path(G=nodeG, source=s, target=d, weight="weight")
    protocol = create_protocol(path, nodeG, f, 1e-10)
    if protocol:
        logger.warning("Found protocol without rate constraint")
        return protocol
    else:
        return None


def main_simulation():
    demand_data = generate_demands()


if __name__ == "__main__":
    main_simulation()