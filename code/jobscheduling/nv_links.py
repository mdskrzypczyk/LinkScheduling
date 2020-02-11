import json


def load_link_data():
    with open("data.json") as f:
        data = json.load(f)

    distances = list(data["rates"].keys())

    d_to_capability = {}
    for d in distances:
        d_to_capability[d] = list(zip(data["fidelities"][d], data["rates"][d]))

    return d_to_capability


if __name__ == "__main__":
    dist_to_cap = load_link_data()
    import pdb
    pdb.set_trace()