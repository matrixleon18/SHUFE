from multiprocessing import Pool
import time
import os


def f(x):
    print("ID: {}  VALUE: {}".format(os.getpid(), x**x))
    time.sleep(x)
    return x**x


def main():
    pool = Pool(processes=10)
    res_list = []
    start_time = time.time()
    for i in range(10):
        res = pool.apply_async(f, [i])
        # res = pool.apply(f, [i])
        res_list.append(res)

    pool.close()
    pool.join()
    print("TIME COST : {}".format(time.time()-start_time))

if __name__ == "__main__":
    main()
