import time

def func1(event1):
    sleepTime = 10
    print("Func1 started!")
    while not(event1.isSet()):
        time.sleep(1)
        print("Hi from func1")

    print(F"Func1 Sleeping {sleepTime}.")
    time.sleep(sleepTime)
    print(F"Func1 Slept {sleepTime}, good bye!")

