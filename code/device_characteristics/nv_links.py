import json


def load_link_data():
    with open("device_characteristics/data.json") as f:
        data = json.load(f)

    distances = list(data["rates"].keys())

    d_to_capability = {}
    for d in distances:
        d_to_capability[d] = list(zip(data["fidelities"][d], data["rates"][d]))

    return d_to_capability


if __name__ == "__main__":
    dist_to_cap = load_link_data()
    for row in zip(*[dist_to_cap[d] for d in sorted(dist_to_cap.keys())]):
        string = ""
        for F, R in row:
            string += " & ({}, {})".format(round(F, 3), round(R, 3))
        print(string)

    import pdb
    pdb.set_trace()