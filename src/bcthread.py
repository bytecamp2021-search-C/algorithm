import threading
import time


class MyThread(threading.Thread):
    def __init__(self, name, function, pos, items, users, tags, group_indices, group_algos, num_recall, tag_set):
        threading.Thread.__init__(self)
        self.name = name
        self.function = function
        self.results = []
        self.pos = pos
        self.items = items
        self.users = users
        self.tags = tags
        self.group_indices = group_indices
        self.group_algos = group_algos
        self.num_recall = num_recall
        self.tag_set = tag_set

    def init(self):
        threading.Thread.__init__(self)

    def run(self):
        self.results = self.function(self.pos, self.items, self.users, self.tags, self.group_indices, self.group_algos,
                                     self.num_recall, self.tag_set)
