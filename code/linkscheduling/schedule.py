class Schedule:
    def __init__(self, schedule_size=100):
        self.jobset = []
        self.schedule_size = schedule_size

    def add_job(self, job, start):
        if self.jobset == []:
            end = start + job.duration
            self.jobset.append((start, end, job))
            job.set_start(start)
        else:
            idx = 0
            for i, job_desc in enumerate(self.jobset):
                s, e, j = job_desc
                if s <= start <= e:
                    raise Exception("Cannot start job at time {}, busy with {}".format(start, j.name))
                elif start + job.duration < s:
                    idx = i
                    break
            self.jobset.insert(idx, (start, start + job.duration, job))

    def num_jobs(self):
        return len(self.jobset)

    def get_completion_time(self):
        if len(self.jobset) == 0:
            return 0
        else:
            last_start, last_end, last_job = self.jobset[-1]
            return last_end

    def find_slot_pre(self, latest_start, duration):
        idx = -1
        for i, job_desc in enumerate(reversed(self.jobset)):
            s, e, j = job_desc
            if latest_start > s:
                continue
            elif latest_start + duration < s


    def find_slot_post(self, earliest_start, duration):



