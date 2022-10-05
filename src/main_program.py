#!~/anaconda3/envs/DLC-GPU/bin/python

import sys , time , logging , faulthandler , pickle , cv2  
import multiprocessing as mp

from upload_ard import uploadIno
from arduino_class import ArduinoHandler
from parsers import typeParser , dirParser , fileParser
from threading import Lock , Thread
from queue import Queue
from threadFuncs import ard_comm , handle_camera , memory_checker
from tk_ard_comm import TkUICommunicator
from directoryTracker import DirTracker
from add_timestamps import add_timestamps_multiple_files
from multiprocessing import Event , Process
from optionHandler import OptionHandler
from verbose_print import verboseprint
from apply_dlc import extract_coords

def main():
    # Only change these if you know what you are doing. 
    # Changing csv might mess things up since we are using the csv library.
    # If fourcc or video_format is changed, the other must be changed accordingly
    # (https://www.fourcc.org/codecs.php)
    program_constants = {   'fourcc':cv2.VideoWriter_fourcc(*'XVID'),
                            'video_format':'avi',
                            'rec_string':'recording',
                            'timestamp_format':'csv',
                            'timestamp_string':'timestamps',
                            'frame_ID_string':'frame_ID',
                            'frame_ID_format':'csv',
                            'channels':1,
                            'cam_fps':550.,
                            'vid_fps':65.,
                            'freq':500,
                            'cam_height':448,
                            'cam_width':448,
                            'cam_offsetX':56,
                            'cam_offsetY':30}

    # Get user input for various parameter settings and choices
    option_handler = OptionHandler(program_constants)
    
    # Ask which script to run
    dir_tracker = DirTracker(option_handler.get_dir_dict())

    # User interface
    user_UI = TkUICommunicator(read_queue, write_queue, proj_queue, ard_ready_event, user_ready, option_handler.get_opt_dict(), dir_tracker) #TODO: Add stimulus queue

    # Get ID of first mouse
    first_ID = user_UI.get_mouse_ID()

    # Create appropriate folders
    dir_tracker.create_new_subfolder(first_ID)

    # Create logfile to save errors
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(dir_tracker.get_current_savepath('errorLog','log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    vprint = option_handler.get_vprint()


    # Upload chosen script to Arduino. Produces alot of debug prints, but nothing to worry about.
    try:
        if option_handler.get_dict_value('doUpload'): uploadIno(dir_tracker, option_handler)
    except Exception as e:
        vprint(e)
        sys.exit()

    # Create Arduino class object to communicate with Arduino and save data.
    ard_object = ArduinoHandler(dir_tracker = dir_tracker, option_handler = option_handler)

    online = False
    # Create new thread for handling and recording camera    
    # (Non-daemon threads are automatically joined at the end of the main script, so we don't have to manually join them, but we do anyways for clarity)
    cam_thread = Thread(target=handle_camera, args=(main_event, cam_handler_ready, user_ready, proj_queue, dir_tracker, option_handler), daemon=False)
    vprint("\n-- Starting cam_thread --\n")
    cam_thread.start()

    # Start new thread for communicating with Arduino
    comm_thread = Thread(target=ard_comm, args=(main_event, cam_handler_ready, ard_ready_event, data_lock, read_queue, ID_queue, write_queue, ard_object, dir_tracker), daemon=False)

    memory_check_thread = Thread(target=memory_checker, args=(read_queue,main_event,10))

    vprint("Main: waiting for cam_handler_ready")
    cam_handler_ready.wait()
    vprint("Main: cam_handler_ready set!")

    vprint("\n-- Starting comm_thread --\n")
    comm_thread.start()
    memory_check_thread.start()
    # Main thread will listen to user input
    while not(main_event.is_set()):

        vprint("Main: waiting for ard_ready_event")
        ard_ready_event.wait()
        vprint("Main: Starting user_UI")
        user_UI.start()
        vprint("Main: Exited user_UI")

        # This event is only cleared if we restart
        if not(ard_ready_event.is_set()):
            new_ID = user_UI.get_mouse_ID()

            ID_queue.put(new_ID)

            user_UI.stop()
            
            if not(user_UI.ask_for_start()):
                main_event.set()
                user_ready.set()
                cam_handler_ready.set()
                time.sleep(1)
                cam_handler_ready.clear()
                break
            
            vprint("Main: User ready!")
            user_ready.set()
        
    try:
        memory_check_thread.join()
        vprint("-- Joining comm_thread --")
        comm_thread.join()
        vprint("-- comm_thread joined --")
        cam_handler_ready.set()
        vprint("-- Joining cam_thread --")
        cam_thread.join()
        vprint("-- cam_thread joined --")
    except RuntimeError:
        vprint("-- cam_thread not started! --")
        pass
    
    
    pickle_savename = dir_tracker.get_current_savepath('tracker','p')

    save_file = open(pickle_savename,'wb')   # Write binary

    # Saving dir_tracker and option_handler for ability to add timestamps at a later stage    
    pickle.dump([dir_tracker.get_all_saved_filenames(),option_handler.get_camera_specs()],save_file)
    vprint("-- Pickle dumped --")

    save_file.close()
    
    # Ask if add timestamps now or later
    if user_UI.query_timestamps(pickle_savename):
        try:
            done_with_timestamps = add_timestamps_multiple_files(dir_tracker.get_all_saved_filenames(), option_handler.get_camera_specs())
            if(done_with_timestamps): user_UI.show_info("Timestamps added!")
        except Exception as e:
            vprint(e)
            user_UI.show_error(e)

    if user_UI.query_dlc(pickle_savename):
        try:
            default_model = '/home/titan/KI2020/DLC/400k_save/config.yaml'
            extract_coords(default_model, pickle_savename)
            user_UI.show_info("DLC data extracted!\n\nTo extract features and create plots use\n\npython dlc_pipeline.py")
        except Exception as e:
            vprint(e)
            user_UI.show_error(e)

    # Apply DLC analysis
    # Save results to file
    
    vprint("\n-- PROGRAM TERMINATED --")
    
    sys.exit()


if __name__ == "__main__":
    faulthandler.enable()

    # Set up events to signal to processes when to wait or proceed
    main_event = Event()
    cam_handler_ready = Event()
    ard_ready_event = Event()
    user_ready = Event()
    user_ready.set()

    # Set up data lock to prevent threads from accessing queue at the same moment
    data_lock = Lock()

    # Set up queue to pass data to and from user input to Arduino
    write_queue = Queue()
    read_queue = Queue()
    proj_queue = mp.Queue()

    # Set up queue to pass new mouse ID to arduino class in separate process
    ID_queue = Queue()
    
    
    try:
        main()
    except Exception as e:
        logging.info(e)