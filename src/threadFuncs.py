#!~/anaconda3/envs/DLC-GPU/bin/python

import time , queue , faulthandler , multiprocessing ,  psutil
import numpy as np
from camera_thread_mp import record_camera
from DisplayHandler import DisplayHandler
from record_fake_camera import record_fake_camera
from projector import Projector
from memory_profiler import profile

def memory_checker(ard_read_queue,main_event,check_interval=60.,warning_thresh=60.,kill_thresh=85.):
    
    init_check_time = 60
    
    time.sleep(init_check_time)

    prev_ram_use = psutil.virtual_memory().percent
    print(f"Initial RAM use: {prev_ram_use}")

    init_t_start = time.time()
    print(f"Checking back in in {init_check_time} seconds")

    time.sleep(init_check_time)

    ram_use = psutil.virtual_memory().percent
    ram_incr = (ram_use - prev_ram_use) # RAM increase per s
    ram_incr_per_s = ram_incr/init_check_time

    print(f'RAM has increased by {ram_incr} MB')
    print(f'Corresponding to {round(ram_incr_per_s,5)} MB increase per second')
    
    if ram_incr_per_s > 0.1:
        sec_until_full = (100 - ram_use)/ram_incr
        print(sec_until_full)
        if sec_until_full < 60*60:
            print("-!- POSSIBLE MEMORY LEAK DETECTED -!-")
            print(f"-!- TITAN MIGHT CRASH IN APPROXIMATELY {round(sec_until_full/60)} MINUTES! -!-")
            print("-!- PROGRAM WILL CONTINUE, BUT PAY ATTENTION TO THE PROMPT -!-")
            print(f"-!- PROGRAM WILL AUTOMATICALLY STOP IF RAM USAGE REACHES {kill_thresh}% -!-")
    
    t_start_memory_check = time.time()
    prev_ram_use = psutil.virtual_memory().percent
    print_initial_warning = True
    
    while not main_event.is_set():
        t_now = time.time()

        if t_now - t_start_memory_check > check_interval:
            ram_use = psutil.virtual_memory().percent
            ram_incr = max(0,(ram_use - prev_ram_use)/check_interval) # RAM increase per s
            
            if ram_use > kill_thresh:
                print("-!- CRITICAL WARNING -!-")
                print(f"-!- RAM USAGE EXCEEDS MAX THRESHOLD {kill_thresh} % -!-")
                print(f"-!- RAM USAGE {ram_use} % -!-")
                print(f"-!- TERMINATING PROGRAM SAFELY -!-")

                ard_read_queue.put('Q')
                break

            if ram_incr > 0:
                sec_until_full = (100 - ram_use)/ram_incr
                
                if ram_use > warning_thresh:
                    check_interval = 10

                    if print_initial_warning:
                        print("-!- WARNING -!-")
                        print(f"-!- RAM USAGE EXCEEDS {warning_thresh}% -!-")
                        print(f"-!- RAM USAGE {ram_use}% -!-")
                        print(f"-!- EXPECTED USAGE ~30% -!-")
                        print(f'-!- RAM INCREASE PER SECOND: {round(ram_incr,2)} -!-')
                        print(f'-!- TIME UNTIL 100%: {round(sec_until_full/60,2)} MINUTES -!-')
                        print_initial_warning = False
                    else:
                        print(f"Safety Memory Check RAM usage: {ram_use}% [WARNING]")
                else:
                    check_interval = init_check_time
                    print_initial_warning = True
                    print(f"Safety Memory Check RAM usage: {ram_use}% [OK]")
                
            prev_ram_use = ram_use
            t_start_memory_check = t_now

def ard_comm(main_event,cam_handler_ready,ard_ready_event,data_lock,read_queue,ID_queue,write_queue,ard,dirTracker):
    """
    Tries to get command from user by queue_in. If this is empty if tries to read from the serial buffer.
    When restarting this is where the DirTracker is updated for this process.
    """
    faulthandler.enable()
    vprint = ard.get_vprint()

    vprint("ard_comm: starting ard.start()")
    ard.start()
    vprint("ard_comm: ard_ready_event.set()")
    ard_ready_event.set()

    # Repeats until user asks to quit
    while( not(main_event.is_set()) ):
        try:
            cmd = write_queue.get_nowait()

            vprint(F"ard_comm: ard.pass_message({cmd})")
            ard.pass_message(cmd)

            if cmd == 'Q':
                vprint("Set main event in ard_comm")
                main_event.set()
                ard.quit_routine(read_queue)

            elif cmd == 'R':
                vprint("ard_comm: calling ard.quit_routine")
                ard.quit_routine(read_queue, restart=True)
                
                vprint("ard_comm: waiting for new_ID from queue...")
                new_ID = ID_queue.get()
                vprint("ard_comm: Got new ID!")

                with data_lock:
                    vprint("ard_comm: Creating new folders")
                    dirTracker.create_new_subfolder(new_ID)
                
                vprint("ard_comm: clearing cam_handler_ready event")
                cam_handler_ready.clear()
                vprint("ard_comm: Waiting for cam_handler_ready")

                # At this point everything is wrapped up
                # Now waiting for processes to restart
                
                cam_handler_ready.wait()
                vprint("ard_comm: cam_handler_ready!")
                
                # GET NEW VARIABLES HERE
                # ARD.RESTART WILL WANT THESE VARS

                ard.restart()
                vprint("ard_comm: Restarted ard")
                ard_ready_event.set()

        except queue.Empty:
            msg = ard.read_process_input()
            if msg: read_queue.put(msg)            
                

@profile(stream=open(f'memory_profiler_handle_camera.log','a+'))
def handle_camera(main_event, cam_handler_ready, user_ready, proj_queue, dirTracker, optHandler):
    """
    Starts, restarts and stops the camera Process when main or restart events are set.
    """
    dirTracker.get_current_savepath
    faulthandler.enable()
    vprint = optHandler.get_vprint()

    cam_function = record_camera if optHandler.get_dict_value('useBlackFly') else record_fake_camera

    cam_recording_ready = multiprocessing.Event()

    spec_dict = optHandler.get_camera_specs()

    img_show_queue = multiprocessing.Queue(1)
    
    online = optHandler.get_dict_value('online')
    print(F"Online tracking: {online}")
    
    # Goes on until user stops
    while not(main_event.is_set()):

        # Handles projected images
        proj = Projector(bgr_col=1,use_fullscreen=True,push_window=4000,rotate=False)
        print("Setting up DisplayHandler")
        
        disp = DisplayHandler(proj, img_show_queue, proj_queue,online)
        print("DisplayHandler setup complete")

        vidName = dirTracker.get_current_savepath(append_str=spec_dict['rec_string'],save_format=spec_dict['video_format'])
        csvName = dirTracker.get_current_savepath(append_str=spec_dict['frame_ID_string'],save_format=spec_dict['frame_ID_format'])
        stimName = dirTracker.get_current_savepath(append_str='stimuli',save_format='csv')

        disp.create_savefile(stimName)

        cam_recording_ready.clear()


        cam_process = multiprocessing.Process(  target=cam_function, 
                                                args=(  img_show_queue, 
                                                        cam_recording_ready, 
                                                        vidName, 
                                                        csvName, 
                                                        spec_dict), 
                                                daemon=False)
        cam_process.start()
 
        vprint("handle_camera: waiting for cam_recording_ready")
        cam_recording_ready.wait()
        vprint("handle_camera: cam ready! setting cam_handler_ready")
        cam_handler_ready.set()

        disp.start_video_stream()
        disp.start_projector()

        if online:
            disp.init_online_processing()

        t_interval = 5
        t_avg = 0
        t_print = time.time()
        i = 0

        print("Started projector")
        # Waits for main or restart events to be set
        while True:
            t_start = time.time()

            disp.update_stream()
            disp.update_projector()

            t_end = time.time()
            
            i += 1
            t_avg += t_end - t_start

            if t_end - t_print > t_interval:
                print(f"Average processing time: {t_avg / i}")
                i = 0
                t_print = time.time()
                t_avg = 0
                    
            if main_event.is_set() or not(cam_handler_ready.is_set()) :
                vprint("handle_camera: main_event or cam_handler_ready is cleared!")
                cam_recording_ready.clear() # Signal camera to stop recording
                vprint("Closing video stream")
                disp.end_stream()
                disp.stop_projector()
                vprint("Video stream closed!")
                vprint("handle_camera: Joining cam_process")
                cam_process.join() # Wrap up recording process
                vprint("handle_camera: Cam pool joined!")
                
                vprint(F"handle_camera: user ready? {user_ready.is_set()}")
                vprint("handle_cmamera: Waiting for user...")
                user_ready.wait()             # Wait for user to restart
                vprint("User ready!")
                break
    
    print("Exiting cam_handler")