from simple_pyspin import Camera
from queue import Queue
import cv2
from PySpin import SpinnakerException
from threading import Event , Thread

def cancel():
    while not(event.is_set()):
        cmd = input("Press Q to cancel:\n>>> ")
        if cmd == "Q":
            event.set()
            break

event = Event()
userThread = Thread(target=cancel, daemon=False)

bf = Camera()
q = Queue()

bf.init()

bf.AcquisitionFrameRateEnable = True
bf.ExposureAuto = 'Off'
bf.GainAuto = 'Off'
bf.GammaEnable = True


bf.TriggerMode = 'On'
bf.TriggerSelector = 'FrameStart'
bf.TriggerSource = 'Line0'
bf.TriggerActivation = 'RisingEdge'
bf.TriggerOverlap = 'ReadOut'
bf.TriggerDelay = 9.
bf.LineSelector = 'Line0'
bf.LineMode = 'Input'

bf.AcquisitionFrameRate = bf.get_info('AcquisitionFrameRate')['max']
bf.ExposureTime = 1000
bf.Gain = 8.5
bf.Gamma = 0.25

bf.start()

i = 0
e = 0
rem = 0
userThread.start()

print("Waiting for camera to start...")
bf.get_array()
print("Got first image!")

while not(event.is_set()):
    try:
        im = bf.get_array(0)
        i+=1
        # q.put(im)
    except SpinnakerException:
        pass
        

print("Images left in buffer?")
qsize1 = i 

try:
    im = bf.get_array(0)
    #q.put(im)
    rem+=1
except SpinnakerException:
    print("Nope!")
else:
    print("Yes")
    while True:
        try:
            im = bf.get_array(0)
            #q.put(im)
            rem+=1
        except SpinnakerException:
            break

userThread.join()

print(F"Got {i*2} imgs!")
print(F"There were {rem} remaining imgs in buffer when stopped. At that point the queue had {qsize1*2} imgs in it.")
