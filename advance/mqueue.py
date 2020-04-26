
from multiprocessing import Process, Queue

import time

def in_q(q):
    for i in ['a','b','c','d','e','f']:
        print("put {} in the queue".format(i))
        q.put(i)
        time.sleep(1)

def out_q(q):
    while True:
        x = q.get(True)
        print("get {} from the queue".format(x))
        time.sleep(2)

def main():
    q = Queue()
    pi = Process(target=in_q, args=[q])
    po = Process(target=out_q, args=[q])
    pi.start()
    po.start()
    po.join()
    po.terminate()

if __name__ == "__main__":
    main()