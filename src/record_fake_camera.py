
def record_fake_camera(cam_recording_ready, video_name, csv_name, cam_specs):
    """
    Generates random matrices (noise) with same dimensions as camera. 
    Used for development purposes when BlackFly is not available.
    Sleeps for 2~ ms to kind of emulate camera speed.
    """
    img_process_ready = multiprocessing.Event()
    
    faulthandler.enable()


    img_proc_queue = multiprocessing.Queue()
    img_show_queue = multiprocessing.Queue(10)

    # Starts a process which will be fed frames from current process and there they are saved to a video.
    n_used_proc = os.cpu_count() - 1
    img_proc_pool = multiprocessing.Pool(n_used_proc, process_imgs, (img_proc_queue, img_process_ready,video_name,csv_name,cam_specs))


    img_show_proc = multiprocessing.Process(target=show_imgs, args=(img_show_queue, img_process_ready), daemon=False)

    print("record_camera: Waiting for img_process_ready to be set")
    img_process_ready.wait()
    print("record_camera: got img_process_ready!")
    cam_recording_ready.set()
    print("record_camera: cam_recording_ready set!")
    img_show_proc.start()
    print("\n-- [VIDEO] RECORDING --\n")
    
    rec_frames = 1
    e = 0
    t_start = time.time()

    # Record until user asks for restart or quit

    while cam_recording_ready.is_set():
        time.sleep(0.0015)
        im = np.random.randint(0,255,(cam_specs['cam_height'],cam_specs['cam_width'],1),dtype='uint8')
        img_proc_queue.put_nowait([im,rec_frames])

        try:
            img_show_queue.put_nowait(im)
            rec_frames += 1
        except queue.Full:
            img_show_queue.qsize()
            pass
    
    dur = time.time() - t_start

    img_process_ready.clear()

    print(F"Recorded for {np.round(dur/60,1)} minutes")
    print(F"Recorded {rec_frames} frames.")
    print(F"Resulting FPS: {rec_frames/dur}")
    print(F"Mean iter time: {dur/(e+rec_frames)*1000} ms")

    img_show_proc.join()
    img_proc_pool.close()
    img_proc_pool.join()

    postprocess_videos(video_name,csv_name,n_used_proc)




def record_camera_OLD(cam_recording_ready,video_name,csv_name,cam_specs):
    """
    Function for recording from BlackFly.
    Gets camera specs from cam_specs (user input in main).

    All calls starting with blackFly.something makes physical changes to setup,
    so they must be updated each time we run the camera.

    Start and stop is triggered by main_event_cam which is shared across multiple threads
    and is set in the getUserInput thread.
    """
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
    
    # One queue for processing images and one for showing
    img_proc_queue = multiprocessing.Queue()
    img_show_queue = multiprocessing.Queue(10)

    # Starts a process which will be fed frames from current process and there they are saved to a video.
    n_used_proc = os.cpu_count() - 1
    img_proc_pool = multiprocessing.Pool(n_used_proc, process_imgs, (img_proc_queue, img_process_ready,video_name,csv_name,cam_specs))

    try:
        img_show_proc = multiprocessing.Process(target=show_imgs, args=(img_show_queue, img_process_ready), daemon=False)
    except Exception as e:
        print(e)

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
    img_show_proc.start()

    print("-- [VIDEO] Waiting for external hardware trigger... --")
    im = blackFly.get_array()
    print("record_camera: got first array!")
    
    print("record_camera: putting frame in queue!")
    img_proc_queue.put([im,0])
    print("record_camera: put frame in queue")

    print("\n-- [VIDEO] RECORDING --\n")
    
    rec_frames = 1
    e = 0
    t_start = time.time()

    # Record until user asks for restart or quit
    while cam_recording_ready.is_set():
        try:
            im = blackFly.get_array(False)
            rec_frames += 1
            img_proc_queue.put_nowait([im,rec_frames])

            blackFly.CounterSelector = 'Counter0'
            curr_count[0] = blackFly.CounterValue
            
            blackFly.CounterSelector = 'Counter1'
            curr_count[1] = blackFly.CounterValue
            
            for i in [0,1]:
                if curr_count[i] < counter[i]: 
                    tot_count[i] += 1
                counter[i] = curr_count[i]
            e += 1

            img_show_queue.put_nowait(im)
            
        except (SpinnakerException , queue.Full):
            pass

    
    dur = time.time() - t_start

    img_process_ready.clear()
    blackFly.stop()

    print(F"Recorded for {np.round(dur/60,1)} minutes")
    print(F"Recorded {rec_frames} frames.")
    print(F"Resulting FPS: {rec_frames/dur}")
    print(F"Mean iter time: {dur/(e+rec_frames)*1000} ms")

    tot_trig_count = tot_count[0]*blackFly.CounterDuration + counter[0] - init_count_trigg
    tot_exp_count = tot_count[1]*blackFly.CounterDuration + counter[1] - init_count_exp

    print(F"Tot trigger count: {tot_count[0]}*{blackFly.CounterDuration} + {counter[0]} - {init_count_trigg} = {tot_trig_count}")
    print(F"Tot exposure count: {tot_count[1]}*{blackFly.CounterDuration} + {counter[1]} - {init_count_exp} = {tot_exp_count}")

    img_show_proc.join()
    img_proc_pool.close()
    img_proc_pool.join()

    postprocess_videos(video_name,csv_name,n_used_proc)
