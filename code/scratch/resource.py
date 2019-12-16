from queue import PriorityQueue


class Resource:
    def __init__(self, name):
        self.name = name
        self.access_queue = PriorityQueue()

    def get_head(self):
        if not self.access_queue.empty():
            return self.access_queue.queue[0]
        else:
            return None

    def add_to_queue(self, priority, item):
        self.access_queue.put((priority, item))

    def pop_from_queue(self):
        if not self.access_queue.empty():
            return self.access_queue.get()
        else:
            return None
