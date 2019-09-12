from netsquid import simutil


class Time:
    def __init__(self, init_time=0):
        self.time = init_time

    def reset(self):
        self.set_time(0)

    def get_time(self):
        return self.time

    def set_time(self, time):
        self.time = time

    def increment_time(self, amount):
        self.time += amount


class SimTime:
    def __init__(self, init_time=0):
        simutil.sim_reset()
        simutil.sim_run(end_time=init_time)

    def reset(self):
        simutil.sim_reset()

    def get_time(self):
        return simutil.sim_time()

    def set_time(self, time):
        simutil.sim_reset()
        simutil.sim_run(end_time=time)

    def increment_time(self, amount):
        simutil.sim_run(duration=amount)
