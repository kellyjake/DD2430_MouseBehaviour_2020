#!~/anaconda3/envs/DLC-GPU/bin/python

import sys , time , csv , multiprocessing , queue , faulthandler , os , re , tqdm , psutil , datetime , pickle
import cv2 as cv
from process_imgs import process_imgs
import numpy as np
import argparse
from memory_profiler import profile

try:
    from PySpin import SpinnakerException
except ModuleNotFoundError:
    pass

try:
    from simple_pyspin import Camera , CameraError # Uncomment this line when using BlackFly on Lab PC!
except ModuleNotFoundError:
    pass


datestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%M%D_%H%M%S')
@profile(stream=open(f'memory_profiler_record_camera.log','a+'))
def record_camera(img_show_queue,cam_recording_ready,video_name,csv_name,cam_specs):
    """
    Function for recording from BlackFly.
    Gets camera specs from cam_specs (user input in main).

    All calls starting with blackFly.something makes physical changes to setup,
    so they must be updated each time we run the camera.

    Start and stop is triggered by main_event_cam which is shared across multiple threads
    and is set in the getUserInput thread.
    """
    queue_check_interval = 10

    img_process_ready = multiprocessing.Event()
    
    faulthandler.enable()

    # Clean up any previous attachments
    try:
        tmp_cam = Camera()
        tmp_cam.init()
        tmp_cam.close()

        blackFly = Camera()
    except (NameError , CameraError):
        print("Camera not connected!")
        sys.exit()

    blackFly.init()

    # Enable parameter editing
    blackFly.AcquisitionFrameRateEnable = True
    blackFly.ExposureAuto = 'Off'
    blackFly.GainAuto = 'Off'
    blackFly.GammaEnable = True

    # Settings for hardware trigger - do not edit
    blackFly.TriggerMode = 'On'
    blackFly.AcquisitionMode = 'Continuous'     
    blackFly.TriggerSelector = 'FrameStart'
    blackFly.TriggerSource = 'Line0'
    blackFly.TriggerActivation = 'RisingEdge'
    blackFly.TriggerOverlap = 'ReadOut'
    blackFly.TriggerDelay = 9.
    blackFly.LineSelector = 'Line0'
    blackFly.LineMode = 'Input'

    # Set FPS to max possible. Acquired framerate will not always be as high as this value.
    blackFly.AcquisitionFrameRate = blackFly.get_info('AcquisitionFrameRate')['max']
    blackFly.ExposureTime = cam_specs['exposure_time']       # us
    blackFly.Gain = cam_specs['gain']                        # db
    blackFly.Gamma = cam_specs['gamma']

    """
    # Setup counters to count missed triggers
    blackFly.CounterSelector = 'Counter0'
    blackFly.CounterEventSource = 'Line0'
    blackFly.CounterEventActivation = 'RisingEdge'
    blackFly.CounterDuration = 65520
    blackFly.CounterTriggerSource = 'Line0'
    blackFly.CounterTriggerActivation = 'RisingEdge'

    init_count_trigg = blackFly.CounterValue

    blackFly.CounterSelector = 'Counter1'
    blackFly.CounterEventSource = 'ExposureStart'
    blackFly.CounterEventActivation = 'RisingEdge'
    blackFly.CounterDuration = 65520
    blackFly.CounterTriggerSource = 'ExposureStart'
    blackFly.CounterTriggerActivation = 'RisingEdge'

    init_count_exp = blackFly.CounterValue

    print(f'Initial TriggerCounter: {init_count_trigg}')
    print(f'Initial ExposureCounter: {init_count_exp}')
    """

    tot_count = [0,0]
    counter = [0,0]
    curr_count = [0,0]

    # Setting dimensions of camera images
    try:
        blackFly.Height = cam_specs['cam_height']
        blackFly.Width = cam_specs['cam_width']
        blackFly.OffsetX = cam_specs['cam_offsetX']
        blackFly.OffsetY = cam_specs['cam_offsetY']
    except CameraError:
        print("Could not set camera height. Current properties:")
        print(F"Height: {blackFly.Height}")
        print(F"Width: {blackFly.Width}")
        print(F"OffsetX: {blackFly.OffsetX}")
        print(F"OffsetY: {blackFly.OffsetY}")

    #Save camera specs for later use
    cam_specs['cam_fps'] = blackFly.AcquisitionResultingFrameRate
    
    # Queue for processing images
    img_proc_queue = multiprocessing.Queue()

    # Starts a process which will be fed frames from current process and there they are saved to a video.
    n_used_proc = os.cpu_count() - 1
    img_proc_pool = multiprocessing.Pool(n_used_proc, process_imgs, (img_proc_queue, img_process_ready,video_name,csv_name,cam_specs))

    print("Starting BlackFly")

    blackFly.start()
        
    print(F"\n-- [VIDEO] SETTINGS --")
    print(F"FPS:\t\t {blackFly.AcquisitionFrameRate}")
    print(F"Processed FPS: {blackFly.AcquisitionResultingFrameRate}")
    print(F"Exposure time:\t{blackFly.ExposureTime}")
    print(F"Gain:\t\t{blackFly.Gain}")
    print(F"Gamma:\t\t{blackFly.Gamma}")

    print("record_camera: Waiting for img_process_ready to be set")
    w_start = time.time()
    img_process_ready.wait()
    print(F"record_camera: got img_process_ready, waited {(time.time() - w_start)*1000} ms!")
    cam_recording_ready.set()
    print("record_camera: cam_recording_ready set!")

    print("-- [VIDEO] Waiting for external hardware trigger... --")
    im = blackFly.get_array()
    print("record_camera: got first array!")
    
    print("record_camera: putting frame in queues!")
    img_proc_queue.put([im,0])
    img_show_queue.put([im,0,time.time()])
    print("record_camera: put frame in queues")


    print("\n-- [VIDEO] RECORDING --\n")
    
    rec_frames = 1
    e = 0
    n = 0
    n_else = 0
    t_start = time.time()
    avg_while = 0
    t_else_avg = 0
    curr_FPS = []
    curr_img_count = 0
    check_FPS_interval = 60
    check_FPS_t = time.time()

    # Record until user asks for restart or quit
    while cam_recording_ready.is_set():
        t_while_start = time.time()
        try:
            im = blackFly.get_array(False)
            t = time.time()
            rec_frames += 1
            img_proc_queue.put_nowait([im,rec_frames])

            """
            # These counters count number of triggers and exposures
            # of the camera to make sure the FPS is around 500.
            # However, introducing this reduces fps to about 400, so 
            # only use for debugging.

            blackFly.CounterSelector = 'Counter0' 
            curr_count[0] = blackFly.CounterValue
            
            blackFly.CounterSelector = 'Counter1'
            curr_count[1] = blackFly.CounterValue
            
            for i in [0,1]:
                if curr_count[i] < counter[i]: 
                    tot_count[i] += 1
                counter[i] = curr_count[i]
            """
            
            t_while_end = time.time()
            

            try:
                img_show_queue.put_nowait([im,rec_frames, t])
            except queue.Full:
                pass

            avg_while += t_while_end - t_while_start
            n += 1
        except SpinnakerException:              # If no new frame - pass
            e += 1
            """
            else:
                t_else_start = time.time()
                                    # Else we want to show the most recent frame
                try:
                    _ = img_show_queue.get_nowait() # Pop previous most recent frame and put new frame into queue    
                except queue.Empty:                 # Unless it is already been showed
                    pass
                finally:
                    try:
                        img_show_queue.put_nowait([im,rec_frames, t])
                    except queue.Full:
                        pass
            """
            #t_else_end = time.time()

            #t_else_avg += t_else_end - t_else_start
            #n_else += 1

        finally:
            check_FPS_t_end = time.time()
            FPS_dur = check_FPS_t_end - check_FPS_t

            if FPS_dur > check_FPS_interval:
                
                frames_this_round = rec_frames - curr_img_count
                print(f"Current FPS: {round(frames_this_round / FPS_dur,1)}")
                check_FPS_t = time.time()
                curr_FPS.append([round(FPS_dur,2),round(frames_this_round / FPS_dur,2)])
                curr_img_count = rec_frames

    
    dur = time.time() - t_start

    img_process_ready.clear()

    try:
        blackFly.stop()
    except SpinnakerException:
        pass

    print(F"\nRecorded for {np.round(dur/60,1)} minutes")
    print(F"Recorded {rec_frames} frames.")
    print(F"Resulting FPS: {rec_frames/dur}")
    print(f'Mean iter time (when get frame): {(avg_while / n)*1000} ms')
    print(F"Mean iter time: {dur/(e+rec_frames)*1000} ms")
    
    """
    tot_trig_count = tot_count[0]*blackFly.CounterDuration + counter[0] - init_count_trigg
    tot_exp_count = tot_count[1]*blackFly.CounterDuration + counter[1] - init_count_exp

    print(F"Tot trigger count: {tot_count[0]}*{blackFly.CounterDuration} + {counter[0]} - {init_count_trigg} = {tot_trig_count}")
    print(F"Tot exposure count: {tot_count[1]}*{blackFly.CounterDuration} + {counter[1]} - {init_count_exp} = {tot_exp_count}")
    """
    
    img_proc_pool.close()
    img_proc_pool.join()

    postprocess_videos(video_name,csv_name,n_used_proc)

    while img_show_queue.qsize() > 0:
        try:
            _ = img_show_queue.get_nowait()
        except queue.Empty:
            pass
        
    FPS_file = os.path.splitext(video_name)[0] + '_FPS_data.p'

    pickle.dump(curr_FPS,open(FPS_file,'wb'))

    print(f"img_proc qsize: {img_proc_queue.qsize()}")
    print(f"img_show qsize: {img_show_queue.qsize()}")

def postprocess_videos(video_name,csv_name,n_used_proc):
    """
    Function to stitch together temporary videos into one full video
    Used in main_program but can be utilized separately too.

    WARNING: removes temporary videos once new full video is created.

    :param video_name:  string of base video name to be stitched together.
                        - example if the videos to be stitched together are
                            video_recording_1_tmp.avi
                            video_recording_2_tmp.avi
                            ...
                            video_recording_N_tmp.avi

                        then input should be 'video_recording.avi'
                        and the endings for each temporary video should be 
                        'k_tmp.avi', where k is a number in the order 1-N.

    :param csv_name:    string of csv file containing indexes for each frame.
                        should be same format as video_name w.r.t. ending, i.e.
                            video_recording_frame_ID_1.tmp.csv
                            video_recording_frame_ID_2.tmp.csv
                            ...
                            video_recording_frame_ID_N.tmp.csv

    :param n_used_proc: number of used processors for creating temporary files. 
                        this is just used to find N, and can be found manually by
                        inspecting the video names and entering the highest number.


    
    """

    faulthandler.enable()

    vid_name_split , vid_format = video_name.split('.')
    csv_name_split , csv_format = csv_name.split('.')

    vid_filenames = []
    csv_filenames = []
    
    vid_readers = []

    csv_files = []

    frame_IDs = []

    n_frames = []

    for i in range(n_used_proc):
        tmp_vid_name = vid_name_split + F"_{i+1}_tmp." + vid_format
        tmp_csv_name = csv_name_split + F"_{i+1}_tmp." + csv_format
    
        vid_filenames.append(tmp_vid_name)
        csv_filenames.append(tmp_csv_name)
        csv_files.append(open(tmp_csv_name,'r'))
        print(tmp_vid_name)
        vid_readers.append(cv.VideoCapture(tmp_vid_name))

        try:
            first_ID = int(re.sub('\\n|','',csv_files[i].readline()))
        except ValueError:
            first_ID = sys.maxsize

        frame_IDs.append(first_ID)
        n_frames.append(vid_readers[i].get(cv.CAP_PROP_FRAME_COUNT))
        print(F"File {i}:\t {n_frames[i]} frames")

    print(F"Total of {sum(n_frames)} frames.")
    
    fourcc = int(vid_readers[0].get(cv.CAP_PROP_FOURCC))
    fps = int(vid_readers[0].get(cv.CAP_PROP_FPS))
    width = int(vid_readers[0].get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid_readers[0].get(cv.CAP_PROP_FRAME_HEIGHT))
    
    vid_writer = cv.VideoWriter(video_name, fourcc, fps, (width,height), isColor=False)
    
    written_frames = 0

    with tqdm.tqdm(total=sum(n_frames),desc='Processing Video',leave=True) as pbar:

        while any(frame_IDs):
            curr_idx = argmin_filter_str(frame_IDs)
            
            success , frame = vid_readers[curr_idx].read()
            
            if success:
                frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                vid_writer.write(frame)
                try:
                    next_id = int(re.sub('\\n|','',csv_files[curr_idx].readline()))
                except ValueError:
                    next_id = ''


                frame_IDs[curr_idx] = next_id
                written_frames +=1
                pbar.update(1)

    for wr,f in zip(vid_readers,csv_files):
        wr.release()
        f.close()

    print(F"Postprocessing done. Wrote {written_frames} frames.")
    vid_writer.release()

    print(F"Removing temporary files")
    for l in [vid_filenames, csv_filenames]:
        for f in l:
            os.remove(f)

    print("Files removed!")

def argmin_filter_str(arr):
    """
    Finds index of smallest value in list or array which may contain
    strings. If so, the index of the smallest non-string value will 
    be returned.
    """
    if not(arr):
        return None
    elif len(arr) == 1:
        return 0
    
    tmp_min = sys.maxsize
    tmp_idx = 0

    for i,val in enumerate(arr):
        if type(val) == str:
            continue

        if val < tmp_min:
            tmp_min = val
            tmp_idx = i

    return tmp_idx

def main(video_name, timestamps_name, n_proc):
    try:
        postprocess_videos(video_name, timestamps_name, n_proc)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    
    example_vid = '/home/titan/KI2020/ExperimentalResults/20200820/Mouse_2/20200820_behaviour2020_iv_2_16/20200820_behaviour2020_iv_2_16_recording.avi'
    example_csv = '/home/titan/KI2020/ExperimentalResults/20200820/Mouse_2/20200820_behaviour2020_iv_2_16/20200820_behaviour2020_iv_2_16_frame_ID.csv'



    # defined command line options
    # this also generates --help and error handling
    CLI=argparse.ArgumentParser()
    
    CLI.add_argument(
        "--video_name",
        type=str
    )
    CLI.add_argument(
        "--timestamps_name",
        type=str  # any type/callable can be used here
    )
    CLI.add_argument(
        "--n_proc",
        type=int
    )
    
    # parse the command line
    args = CLI.parse_args()
    # access CLI options

    video_name = args.video_name
    timestamps_name = args.timestamps_name
    n_proc = args.n_proc

    main(video_name,timestamps_name,n_proc)