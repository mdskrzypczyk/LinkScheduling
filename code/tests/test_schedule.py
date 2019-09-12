import unittest
from linkscheduling.job import Job
from linkscheduling.schedule import Schedule, DEFAULT_SCHEDULE_SIZE, DEFAULT_SLOT_SIZE


class TestSchedule(unittest.TestCase):
    def test_init(self):
        # Check the Schedule ID system
        firstID = Schedule.sched_ID
        schedule = Schedule()

        # Check Schedule initialization
        clock = Schedule.clock
        self.assertEqual(schedule.get_schedule_ID(), firstID)
        self.assertEqual(schedule.get_num_total_slots(), DEFAULT_SCHEDULE_SIZE)
        self.assertEqual(schedule.get_current_slot_time(), clock.get_time())
        self.assertEqual(schedule.get_slot_size(), DEFAULT_SLOT_SIZE)
        self.assertEqual(schedule.get_availability_view(), [0 for _ in range(DEFAULT_SCHEDULE_SIZE)])
        self.assertEqual(schedule.get_job_view(), [None for _ in range(DEFAULT_SCHEDULE_SIZE)])
        self.assertEqual(schedule.job_lookup, {})

    def test_get_availability_view(self):
        schedule = Schedule()
        view = schedule.get_availability_view()
        self.assertEqual(view, [schedule.SLOT_AVAILABLE for _ in range(DEFAULT_SCHEDULE_SIZE)])
        window_size = 10
        view = schedule.get_availability_view(window_size=window_size)
        self.assertEqual(view, [schedule.SLOT_AVAILABLE for _ in range(window_size)])

        # Check view after inserting stuff into the availability list

    def test_get_job_view(self):
        schedule = Schedule()
        view = schedule.get_job_view()
        self.assertEqual(view, [None for _ in range(DEFAULT_SCHEDULE_SIZE)])
        window_size = 10
        view = schedule.get_job_view(window_size=window_size)
        self.assertEqual(view, [None for _ in range(window_size)])

        # Check view after inserting stuff into the job list

    def test_insert_job(self):
        schedule = Schedule()

        job_name = "TestJob"
        job_ID = 42
        number_slots = 5
        job_duration = number_slots * schedule.get_slot_size()
        new_job = Job(name=job_name, ID=job_ID, duration=job_duration)

        view = schedule.get_availability_view()
        self.assertEqual(view, [schedule.SLOT_AVAILABLE for _ in range(DEFAULT_SCHEDULE_SIZE)])

        view = schedule.get_job_view()
        self.assertEqual(view, [None for _ in range(DEFAULT_SCHEDULE_SIZE)])

        start_slot = 3
        job_start = start_slot * schedule.get_slot_size()
        schedule.insert_job(job=new_job, start_time=job_start, duration=job_duration)

        view = schedule.get_availability_view()
        expected_view = [schedule.SLOT_AVAILABLE for _ in range(start_slot)] +\
                        [schedule.SLOT_OCCUPIED for _ in range(start_slot, start_slot + number_slots)] +\
                        [schedule.SLOT_AVAILABLE for _ in range(start_slot + number_slots, DEFAULT_SCHEDULE_SIZE)]
        self.assertEqual(view, expected_view)

        view = schedule.get_job_view()
        expected_view = [None for _ in range(start_slot)] +\
                        [(job_start, job_duration, new_job) for _ in range(start_slot, start_slot + number_slots)] +\
                        [None for _ in range(start_slot + number_slots, DEFAULT_SCHEDULE_SIZE)]
        self.assertEqual(view, expected_view)

    def test_remove_job(self):
        schedule = Schedule()

        job_name = "TestJob"
        job_ID = 42
        number_slots = 5
        job_duration = number_slots * schedule.get_slot_size()
        new_job = Job(name=job_name, ID=job_ID, duration=job_duration)
        start_slot = 3
        job_start = start_slot * schedule.get_slot_size()
        schedule.insert_job(job=new_job, start_time=job_start, duration=job_duration)

        removed_job = schedule.remove_job(job_ID=job_ID)

        self.assertEqual(new_job, removed_job)

        view = schedule.get_availability_view()
        self.assertEqual(view, [schedule.SLOT_AVAILABLE for _ in range(DEFAULT_SCHEDULE_SIZE)])

        view = schedule.get_job_view()
        self.assertEqual(view, [None for _ in range(DEFAULT_SCHEDULE_SIZE)])

    def test_roll_forward(self):
        pass
