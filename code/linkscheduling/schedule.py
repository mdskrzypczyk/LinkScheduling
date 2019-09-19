import numpy as np
from collections import defaultdict
from log import LSLogger
from math import ceil
from util import Time, SimTime

DEFAULT_SCHEDULE_SIZE = 100
DEFAULT_SLOT_SIZE = 100000


def roll_schedules_forward(schedules):
    for schedule in schedules:
        schedule.roll_forward()


class Schedule:
    # Common clock to all schedules
    clock = Time()

    # Track Schedule ID
    sched_ID = 0

    # Availability encoding
    SLOT_AVAILABLE = 0
    SLOT_OCCUPIED = 1

    def __init__(self, schedule_size=DEFAULT_SCHEDULE_SIZE, slot_size=DEFAULT_SLOT_SIZE):
        self.ID = self.sched_ID
        self.name = "Schedule-{}".format(self.sched_ID)
        self.sched_ID += 1

        # Add a logger
        self.logger = LSLogger(name="{}-Logger".format(self.name))

        # The number of slots that are kept track of in the schedule
        self.schedule_size = schedule_size

        # Track the amount of time each slot corresponds to (in nanoseconds)
        self.slot_size = slot_size

        # Track what time the head slot corresponds to
        self.head_time = self.clock.get_time()

        # A bitmap indicating if a job occupies the indicated slot
        self.availability = [self.SLOT_AVAILABLE for _ in range(self.schedule_size)]

        # List of the job IDs in this schedule
        self.jobs = [None for _ in range(self.schedule_size)]

        # ID to job lookup table
        self.job_lookup = defaultdict()

    def get_slot_size(self):
        return self.slot_size

    def get_num_total_slots(self):
        return self.schedule_size

    def get_current_slot_time(self):
        return self.head_time

    def get_schedule_ID(self):
        return self.ID

    def get_availability_view(self, start_slot=None, window_size=None):
        # Set the view window size
        if window_size is None or window_size > self.schedule_size:
            window_size = self.schedule_size

            # Allow checking from different points in the view
            if start_slot is None:
                start_slot = 0

            # If start slot out of scope then error
            elif start_slot > self.schedule_size:
                self.logger.error("Availability view starting from slot {} outside of range {}".format(start_slot,
                                                                                                       self.schedule_size))

        # Get a list of the availability information for the specified slots
        availability_view = self.availability[start_slot:window_size]

        return availability_view

    def get_continuous_slots(self, min_size=1):
        continuous_slots = []
        for i in range(self.schedule_size):
            if self.availability[i:i+min_size] == [self.SLOT_AVAILABLE for _ in range(min_size)]:
                continuous_slots.append((i, i+min_size-1))

        return continuous_slots

    def get_job_view(self, start_slot=None, window_size=None):
        # Set the view window size
        if window_size is None or window_size > self.schedule_size:
            window_size = self.schedule_size

        # Allow checking from different points in the view
        if start_slot is None:
            start_slot = 0

        # If start slot out of scope then error
        elif start_slot > self.schedule_size:
            self.logger.error("Job view starting from slot {} outside of range {}".format(start_slot,
                                                                                                   self.schedule_size))

        # Get a list of the job information for the specified slots
        job_view = [self.replace_with_name(self.job_lookup[job_ID]) if job_ID is not None else None
                    for job_ID in self.jobs[start_slot:window_size]]

        return job_view

    def replace_with_name(self, job_tuple):
        s, d, j = job_tuple
        return np.round(s, decimals=3), np.round(d, decimals=3), j.name

    def insert_job(self, job, start_time, duration):
        # Check if the start time is in the past
        if start_time < self.head_time:
            self.logger.error("Attempted to insert job {} into {} "
                              "at time {} when current time is {}!".format(job, self.name, start_time,  self.head_time))
            return False

        # Check if the job has positive duration
        if duration <= 0:
            self.logger.error("Attempted to insert job {} with duration {} into {}!".format(job, duration, self.name))
            return False

        # Convert to slot information
        time_diff = start_time - self.head_time
        starting_slot = int(np.round(time_diff / self.slot_size))
        num_occupied_slots = ceil(duration / self.slot_size)
        job_ID = job.get_ID()

        # Set the availability and job information in the schedule
        for slot_num in range(starting_slot, starting_slot + num_occupied_slots):
            self.availability[slot_num] = self.SLOT_OCCUPIED
            self.jobs[slot_num] = job_ID

        # Add the job to lookup
        self.job_lookup[job_ID] = (start_time, duration, job)
        return True

    def remove_job(self, job_ID):
        # Check if the job is in the schedule anymore
        if job_ID not in self.job_lookup.keys():
            self.logger.debug("Job {} not found in {}".format(job_ID, self.name))
            return False

        # Get the job information and convert it into slot information
        start_time, duration, job = self.job_lookup.pop(job_ID)
        time_diff = start_time - self.head_time
        starting_slot = max(0, time_diff // self.slot_size)
        final_slot = ceil((time_diff + duration) / self.slot_size)

        # Clear availability and job slots that were removed
        for slot_num in range(starting_slot, final_slot):
            self.availability[slot_num] = self.SLOT_AVAILABLE
            self.jobs[slot_num] = None

        # Return the removed job
        return job

    def roll_forward(self):
        self.logger.debug("Rolling forward {}".format(self.name))

        # Calculate how much time has passed
        curr_time = self.clock.get_time()
        time_diff = curr_time - self.head_time

        # If not time has passed do nothing
        if time_diff == 0:
            self.logger.debug("Time has not changed")
            return None

        # Negative time passing should not be allowed
        elif time_diff < 0:
            self.logger.exception("Time has rolled backward!  Used to be {}, is now {}".format(self.head_time,
                                                                                               curr_time))
            return None

        # If some time has passed, update the availability and job list in this schedule
        else:
            # Get the number of slots that have passed
            num_old_slots = time_diff // self.slot_size

            # Get availability information for the timeslots that elapsed
            old_availability = self.availability[:num_old_slots]

            # Update current availability
            self.availability = self.availability[num_old_slots:] + [self.SLOT_AVAILABLE for _ in range(num_old_slots)]

            # Get the IDs of the jobs that occurred in the elapsed time slots
            old_job_IDs = self.jobs[:num_old_slots]

            # Update the current set of jobs
            self.jobs = self.jobs[num_old_slots:] + [None for _ in range(num_old_slots)]

            # Filter down to the unique job IDs
            filtered_IDs = list(filter(lambda ID: ID is not None, old_job_IDs))

            # Track which IDs no longer belong to schedule after rolling forward
            removed_IDs = []

            # Log info about when the timeslots happened
            slot_time = self.head_time
            for ID in set(filtered_IDs):
                # Get the job
                job_info = self.job_lookup[ID]
                start_time, duration, job = job_info

                # Check if the job is set to finish by the updated time and remove if so
                if start_time + duration <= curr_time:
                    self.logger.debug("Removing job {} from {} at slot time {}".format(job, self.name, slot_time))
                    removed_IDs.append(ID)

                # Update the current tracked slot time
                slot_time += self.slot_size

            # Remove job information for the jobs that have completed in the elapsed time
            for ID in set(removed_IDs):
                self.job_lookup.pop(ID)

            # Update the timestamp of the head timeslot
            self.head_time = (curr_time // self.slot_size) * self.slot_size

            # Return elapsed schedule information (could be useful)
            return old_availability, filtered_IDs


class LinkSchedule(Schedule):
    def __init__(self, availability_view):
        super(LinkSchedule, self).__init__()
        self.availability = availability_view
