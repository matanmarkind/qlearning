import sys, os

parent_dir = os.path.dirname(sys.path[0])
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)
from utils.RingBuffer import RingBuf


if __name__ == '__main__' and __package__ is None:
    print(sys.path)
    print(RingBuf)
