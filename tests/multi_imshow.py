import cv2
import multiprocessing as mp
import threading
import numpy as np
import time
from projector import Projector , Stimuli , TriggerEllipse

def test_mp(wname):
    print("In test_mp")
    proj = Projector(0)
    print("Created projector")
    stim = Stimuli(proj)
    print("Created stim")

    for i,x in enumerate(np.linspace(-.3,.3,3)):
        for j,y in enumerate(np.linspace(-.3,.3,3)):
            proj.add_trigger((x,y),(70,20),i*3+j)
            print("Added trigger")
    
    #proj.show_triggers(0.5)


    w = cv2.namedWindow('test')
    imgs = []
    for i in range(100):
        img = np.random.random((40,40))
        cv2.imshow('test',img)

wname = 'test'
p = mp.Process(target=test_mp,args=(wname,))


#w2 = cv2.namedWindow(wname)

p.start()
print("Main showing img")
#cv2.imshow(w2,img2)
#cv2.waitKey(1000)

p.join()