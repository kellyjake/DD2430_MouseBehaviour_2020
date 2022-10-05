import numpy as np
import cv2 , sys , os
from time import time , sleep
from multiprocessing import Event , Queue , Pool , Process
from queue import Full
from procc_imgs_test import proc_imgs_pool , show_imgs
try:
    from simple_pyspin import Camera
    from PySpin import SpinnakerException
except ImportError:
    pass

from tqdm import tqdm


def main():
    try:    
        bf = Camera()
        bf.init()
    except:
        pass

    try:
        bf.UserSetSelector = 'UserSet0'
        bf.UserSetLoad()
    except:
        print("Couldn't load user settings")
        pass

    try:
        bf.start()
    except:
        pass
    
    filename = os.path.join('tests','test_imgs','testvid')
    ev = Event()
    proc_q = Queue()
    show_q = Queue(10)
    p_list = []

    img_proc = Process(target=proc_imgs_pool, args=(proc_q,ev,filename), daemon=False)
    img_proc.start()
    
    #img_show = Process(target=show_imgs, args=(show_q,ev))
    #img_show.start()

    
    print("Creating pool")
    
    pool = Pool(os.cpu_count() - 2, proc_imgs_pool, (proc_q,ev,filename))

    t_tot_start = time()
    t = 0
    dur = 0
    q_size = proc_q.qsize()
        
    for _ in range(10000):
        try:
            t_start = time()
            #im = bf.get_array(False)
            im = np.random.randint(0,255,(480,480,1),dtype='uint8')
            
            proc_q.put_nowait([im,t])
            try:
                show_q.put_nowait(im)
            except Full:
                pass
            
            
            t += 1
            dur += time() - t_start
        
        except SpinnakerException:
            pass
    
    t_tot_end = time() - t_tot_start
    #cv2.destroyWindow(wName)
    print("Closing cam")
    try:
        bf.close()
    except:
        pass

    print("Waiting for queues to empty")
    while proc_q.qsize() | show_q.qsize():
        pass
    
    print("Set event")
    ev.set()
    
    try:
        #print("Closing pool")
        pool.close()
        #print("Joining pool")
        pool.join()
    except NameError:
        pass
        
    try:
        img_proc.join()
        img_show.join()
    except NameError:
        pass

    print("Closing show_q")
    

    #show_q.close()
    #proc_q.close()

    print(F"Total duration {t_tot_end}")
    print(F"Took {dur/t*1000} ms to check for img")
    

if __name__ == "__main__":
    main()