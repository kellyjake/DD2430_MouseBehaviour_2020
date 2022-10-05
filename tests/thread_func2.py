import time 

def func2(event2):
    sleepTime = 5
    print("Func2 started!")
    while not(event2.isSet()):
        time.sleep(1)
        print("Hi from func2")

    print(F"Func2 Sleeping {sleepTime}.")
    time.sleep(sleepTime)
    print(F"Func2 Slept {sleepTime}, good bye!")

