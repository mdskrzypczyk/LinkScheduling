class Job:
    def __init__(self, name, ID, duration=0, params=None):
        self.name = name
        self.ID = ID
        self.duration = duration
        self.params = params

    def get_ID(self):
        return self.ID

    def get_duration(self):
        return self.duration


class JobOrdering:
    def __init__(self, size):
        self.size = size
        self.ordering = [None for _ in range(size)]

    def __copy__(self):
        jo = JobOrdering(self.size)
        jo.ordering = [_ for _ in self.ordering]
        return jo

    def set_rel_start_post(self, index):
        self.ordering[index] = True

    def set_rel_start_pre(self, index):
        self.ordering[index] = False

    def get_id(self):
        id_string = ''
        for c in self.ordering:
            if c is None:
                id_string += 'x'
            elif c is True:
                id_string += '1'
            elif c is False:
                id_string += '0'
            else:
                raise Exception('Invalid item in ordering!')

        return id_string
