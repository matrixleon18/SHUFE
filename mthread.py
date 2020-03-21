
from threading import Thread
import time


def timewrapper(func):
    def timecount():
        start_time = time.time()
        func()
        end_time = time.time()
        print("TIME COST : {}".format(end_time-start_time))

    return timecount


def my_counter():
    i = 0
    for _ in range(100000000):
        i = i+1
    return True

@timewrapper
def main():
    thread_array = {}

    for tid in range(2):
        t =Thread(target=my_counter)
        t.start()
        thread_array[tid] = t
    for i in range(2):
        thread_array[i].join()


@timewrapper
def main2():
    my_counter()


if __name__ == "__main__":
    main()
    main2()