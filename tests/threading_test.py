from threading import Event , Thread
import time

from thread_func1 import func1
from thread_func2 import func2

#def main():
if __name__ == '__main__':
    event = Event()
    thread1 = Thread(target=func1, args=[event], daemon=False)
    thread2 = Thread(target=func2, args=[event], daemon=False)

    thread1.start()
    thread2.start()

    time.sleep(3)
    event.set()

"""
if __name__ == '__main__':
    main()
    """