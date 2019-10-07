from multiprocessing import Pool, cpu_count, Value, set_start_method, Manager, Process
import time
from concurrent.futures import ProcessPoolExecutor
from ctypes import c_int
import platform


def test_method(queue):
    print("test start")
    time.sleep(1)
    queue.put(1)
    print("test end")


def main():
    process_count = cpu_count()
    m = Manager()
    queue = m.Queue()
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for i in range(process_count):
            executor.submit(test_method, queue)

        if queue.qsize() > process_count - 1:
            print("value is reached max value!")


def main2():
    process_count = cpu_count()
    m = Manager()
    queue = m.Queue()
    p_list = []
    for i in range(process_count):
        p = Process(target=test_method, args=(queue,))
        p.start()
        p_list.append(p)

    while True:
        if queue.qsize() > process_count - 1:
            print("value is reached max value!")
            break

    for p in p_list:
        p.join()
        p.terminate()


if __name__ == "__main__":
    # fix bug of macOS
    if platform.system() == 'Darwin':
        set_start_method('forkserver')

    # main()
    main2()
