class Job:
    def __init__(self, name, duration=0, params=None):
        self.name = name
        self.duration = duration
        self.params = params
        self.start = None

    def get_duration(self):
        return self.duration

    def set_start(self, start):
        self.start = start

    def get_start(self):
        if self.start is None:
            raise Exception("Job {} has not been assigned a start time!".format(self.name))
        return self.start

    def get_end(self):
        if self.start is None:
            raise Exception("Job {} has not been assigned a start time!".format(self.name))
        return self.start + self.duration       


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
