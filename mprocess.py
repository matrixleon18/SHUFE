
from multiprocessing import Process
import time

def timewrapper(func):
    def timecount():
        starttime = time.time()
        func()
        endtime = time.time()
        print(endtime-starttime)
    return timecount

def my_counter():
    i = 0
    for _ in range(100000000):
        i = i+1
    return True

def f(n):
    time.sleep(2)
    print(n*n)

@timewrapper
def main():
    for i in range(10):
        # p = Process(target=f, args=[i,])
        p = Process(target=my_counter)
        p.start()

if  __name__ == "__main__":
    main()

