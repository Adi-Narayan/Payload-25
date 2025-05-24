# diskqueue.py
import os
import pickle
from pathlib import Path

class DiskQueue:
    def __init__(self, directory="pose_spill"):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0  # write index
        self.read_counter = 0  # read index

    def put(self, item):
        filename = self.dir / f"{self.counter:08d}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(item, f)
        self.counter += 1

    def get(self):
        filename = self.dir / f"{self.read_counter:08d}.pkl"
        if filename.exists():
            with open(filename, 'rb') as f:
                item = pickle.load(f)
            os.remove(filename)
            self.read_counter += 1
            return item
        else:
            return None

    def qsize(self):
        return max(0, self.counter - self.read_counter)

    def empty(self):
        return self.qsize() == 0
