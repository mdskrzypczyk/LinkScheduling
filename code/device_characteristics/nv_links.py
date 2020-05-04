import json


def load_link_data():
    """
    Loads the link capability information
    :return: type dict
        Dictionary of link lengths to lists of link capabilities
    """
    with open("device_characteristics/data.json") as f:
        data = json.load(f)

    with open("device_characteristics/data.json", "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)

    distances = list(data["rates"].keys())

    d_to_capability = {}
    for d in distances:
        d_to_capability[d] = list(zip(data["fidelities"][d], data["rates"][d]))

    return d_to_capability


if __name__ == "__main__":
    dist_to_cap = load_link_data()
    lengths = [5*i for i in range(1,11)]
    for l in lengths:
        cap = dist_to_cap[str(l)]
        string = "${}km$".format(l)
        for F, R in reversed(sorted(cap)):
            string += " & {}".format(round(R, 1))
        print(string)
