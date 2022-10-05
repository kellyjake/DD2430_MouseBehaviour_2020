from simple_pyspin import Camera
from PySpin import SpinnakerException
from queue import Full , Empty
import time , cv2 , sys , serial , argparse
import multiprocessing as mp
from procc_imgs_test import procc_imgs
from clear_buffer import clear_buffer
from verbose_print_test import VerbosePrint
from numpy import floor

def test_cam_speed(freq,t):
    verbose = VerbosePrint('cam_speed_test.log')
    vprint = verbose.get_vprint()

    ev = mp.Event()
    q = mp.Queue()


    proc = mp.Process(target=procc_imgs, args=(ev, q, t), daemon=False)

    bf = Camera()
    bf.init()
    try:
        bf.UserSetSelector = 'UserSet0'
        bf.UserSetLoad()
    except SpinnakerException:
        pass
    
    bf.AcquisitionFrameRateEnable = True
    bf.ExposureAuto = 'Off'
    bf.GainAuto = 'Off'

    bf.AcquisitionMode = 'Continuous'
    bf.TriggerMode = 'On'
    bf.TriggerSelector = 'FrameStart'
    bf.TriggerSource = 'Line0'
    bf.TriggerActivation = 'RisingEdge'
    bf.TriggerOverlap = 'ReadOut'
    bf.TriggerDelay = 9.
    bf.LineSelector = 'Line0'
    bf.LineMode = 'Input'

    # Setup counters to catch missed triggers

    bf.CounterSelector = 'Counter0'
    init_count_trigg = bf.CounterValue

    bf.CounterEventSource = 'Line0'
    bf.CounterEventActivation = 'RisingEdge'
    bf.CounterDuration = 65520
    bf.CounterTriggerSource = 'Line0'
    bf.CounterTriggerActivation = 'RisingEdge'

    bf.CounterSelector = 'Counter1'
    init_count_exp = bf.CounterValue

    bf.CounterEventSource = 'ExposureStart'
    bf.CounterEventActivation = 'RisingEdge'
    bf.CounterDuration = 65520
    bf.CounterTriggerSource = 'ExposureStart'
    bf.CounterTriggerActivation = 'RisingEdge'
    
    
    
    vprint(F"Initial trigger counter: {init_count_trigg}")
    vprint(F"Initial exposure counter: {init_count_exp}")

    vprint(F"Given frequency: {freq}")
    bf.AcquisitionFrameRate = min(bf.get_info('AcquisitionFrameRate')['max'],freq + 5)
    vprint(F"Framerate set to {bf.AcquisitionFrameRate}")


    bf.ExposureTime = 1000
    bf.Gain = 10
    bf.Gamma = 0.25

    counter = [init_count_trigg , init_count_exp]
    tot_count = [0,0]
    curr_count = [0,0]

    e = 0
    j = 1

    vprint("Trying to start proc")
    proc.start()
    vprint("Started proc")
    
    bf.CounterSelector = 'Counter0'
    vprint("Starting cam")
    bf.start()
    vprint("Started cam")
    vprint("Waiting for hardware trigger")
    im = bf.get_array(True)
    vprint("Got first frame")
    start_t = time.time()

    while not(ev.is_set()):
        try:
            im = bf.get_array(False)
            q.put_nowait(im)
            j += 1
        except SpinnakerException:
            bf.CounterSelector = 'Counter0'
            curr_count[0] = bf.CounterValue
            
            bf.CounterSelector = 'Counter1'
            curr_count[1] = bf.CounterValue
            
            for i in [0,1]:
                if curr_count[i] < counter[i]: 
                    tot_count[i] += 1
                counter[i] = curr_count[i]
            
            e += 1
            
            
            
            
            



    end_t = time.time()

    dur = end_t - start_t
    

    # Resets counter
    bf.CounterSelector = 'Counter0'
    bf.CounterTriggerSource = 'Off'
    bf.CounterResetSource = 'Line0'
    bf.CounterResetActivation = 'RisingEdge'        

    bf.CounterSelector = 'Counter1'
    bf.CounterTriggerSource = 'Off'
    bf.CounterResetSource = 'Line0'
    bf.CounterResetActivation = 'RisingEdge'        
    vprint("Counters reset!")
    
    # Show stats
    tot_trig_count = tot_count[0]*bf.CounterDuration + counter[0] - init_count_trigg
    tot_exp_count = tot_count[1]*bf.CounterDuration + counter[1] - init_count_exp    
    missed_frames = tot_trig_count - j + 1

    vprint(F"\nDuration: {dur}")
    vprint(F"Tot trigger count: {tot_count[0]}*{bf.CounterDuration} + {counter[0]} - {init_count_trigg} = {tot_trig_count}")
    vprint(F"Tot exposure count: {tot_count[1]}*{bf.CounterDuration} + {counter[1]} - {init_count_exp} = {tot_exp_count}")
    vprint(F"Got {j} images from BlackFly (tot exposure count)")
    try:
        vprint(F"Missed {missed_frames} ({round(missed_frames/tot_trig_count*100,3)}%)")
    except ZeroDivisionError:
        pass

    vprint(F"Average iter time: {dur/(e+j)*1000} ms")

    vprint(F"Expected no. frames (freq*dur): {floor(freq*dur)}")

    vprint(F"Resulting framerate: {j/dur}")

    vprint(F"No. of empty frames: {e}")
    

    bf.stop()
    proc.join()
    
    vprint("#################################### END ####################################\n")


if __name__ == "__main__":

    # defined command line options
    # this also generates --help and error handling
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
    "--freqlist",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=int,
    default=[500],  # default if nothing is provided
    )
    CLI.add_argument(
    "--durationlist",
    nargs="*",
    type=int,  # any type/callable can be used here
    default=[60],
    )

    # parse the command line
    args = CLI.parse_args()
    # access CLI options

    freq = args.freqlist
    times = args.durationlist

    print(F"Frequencies: {freq}")
    print(F"Durations: {times}")
    
    for f , t in zip(freq,times):
        print(F"Running test_cam_speed({f},{t})")
        test_cam_speed(f,t)
