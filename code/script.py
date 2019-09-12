from linkscheduling.job import Job
from linkscheduling.schedule import Schedule


def increment_schedule_time(schedules):
    for schedule in schedules:
        schedule.roll_forward()


def main():
    schedule = Schedule()
    clock = schedule.clock
    timeslot_duration = schedule.get_slot_size()
    new_job = Job(name="TestJob", ID=42, duration=5*timeslot_duration)

    curr_time = clock.get_time()
    schedule.insert_job(job=new_job, start_time=curr_time, duration=new_job.duration)

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()
