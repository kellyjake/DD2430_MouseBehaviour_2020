import cv2 , logging , csv
from queue import Empty
from tqdm import tqdm
from time import sleep , time
import multiprocessing as mp
import faulthandler

def process_imgs(in_queue,img_process_ready,video_name,csv_name,spec_dict):
    """
    Gets images put in the queue by recordCamera() in camera_thread_mp.py.
    Saves images to a video with the formatting given by the spec_dict (OptionHandler.get_camera_specs())
    """
    
    this_proc_ID = "_" + mp.current_process().name[-1] + "_tmp"

    vid_name_split , vid_format = video_name.split('.')
    csv_name_split , csv_format = csv_name.split('.')

    vid_name_split += this_proc_ID
    csv_name_split += this_proc_ID

    new_vid_name = '.'.join([vid_name_split,vid_format])
    new_csv_name = '.'.join([csv_name_split,csv_format])

    csv_file = open(new_csv_name,'w')

    print(F"process_imgs: worker {this_proc_ID} writing to {new_vid_name}")

    vidWriter = cv2.VideoWriter(new_vid_name,spec_dict['fourcc'], spec_dict['vid_fps'], (spec_dict['cam_width'],spec_dict['cam_height']), isColor=False)
    csvWriter = csv.writer(csv_file)

    img_process_ready.set()
    
    while img_process_ready.is_set() or in_queue.qsize() > 0:
        try:
            im , t = in_queue.get_nowait()
            vidWriter.write(im)
            csvWriter.writerow([t])
        except Empty:
            pass
        except Exception as e:
            print("Error occured!")
            print(e)
    
    vidWriter.release()
    csv_file.close()

    if in_queue.qsize():
        print(F"Process_imgs: Warning! Remaining imgs in queue: {in_queue.qsize()}")


def show_imgs(show_queue,event):
    faulthandler.enable()

    timer = time()
    disp_interval = 10

    wName = 'Video Stream'
    _ = cv2.namedWindow(wName)
    
    print("Image display ready")

    while event.is_set() or show_queue.qsize():
        try:
            im = show_queue.get_nowait()    
            cv2.imshow(wName,im)
            cv2.waitKey(1)
            time_now = time()
            if (time_now - timer) > disp_interval:
                timer = time()
                print(f'Queue size for showing imgs: {show_queue.qsize()}')

        except Empty:
            pass
    
    print("Destroying window!")
    print(f"Queue size: {show_queue.qsize()}")
    print(f"Event set?: {event.is_set()}")

    cv2.destroyWindow(wName)
