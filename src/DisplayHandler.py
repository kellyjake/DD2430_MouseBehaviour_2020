#!~/anaconda3/envs/DLC-GPU/bin/python
import time , csv ,  queue , faulthandler , tqdm
import cv2 as cv
import numpy as np
from dlclive import DLCLive , Processor
from kalman_filter import create_kalman_filters , predict_and_update
import tensorflow as tf
from pose2d import get_output_variables_2d , gaze_vec , get_midpt , h_angle_x
from skimage import draw

class DisplayHandler():

    def __init__(self,projector, stream_queue, projector_queue, online=False):
        self.projector = projector
        self.stream_queue = stream_queue
        self.projector_queue = projector_queue
        self.__online = online
        
        self.__config_path = '/home/titan/KI2020/DLC/KI2020_Training-Valter-2020-09-19/exported-models/DLC_KI2020_Training_resnet_50_iteration-0_shuffle-1/'

        self.__stream_name = 'Video Stream'
        self.__empirical_mean = 0  # Mean error (ms) between estimated and 
                                    # observed stimuli presentaiton time.
                                    # Std dev = 6.47
                                    
        self.__empirical_error = 0    # (90% is within +- 8.4 ms)

        self.p2_time = 0

        self.__current_midpt = np.array([0,0])
        self.__current_midpt_unit = np.array([0,0])
        self.__current_midpt_proj_coords = np.array([0,0])
        self.__current_gaze_vector = np.array([0,0])
        self.__current_gaze_angle = 0
        self.__coords = np.zeros((4,2))

        self.__current_snout_pos = np.array([0,0])
        self.__current_snout_pos_unit = np.array([0,0])
        self.__current_snout_proj_coords = np.array([0,0])
        self.__filtered_coords = np.zeros((4,2))

        self.__threshold = 0.95
        self.bgr_col = 0

        self.__process_img_func = lambda im : self.display_stream_img(im)
        
        self.__clean_coords_tmp = lambda im : im.T # Before init of kalman

        self.__kalman_initialized = False
            


    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    #*#*#*#*#*#*# PUBLIC METHODS *#*#*#*#*#*#*#
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

    def start_video_stream(self):
        self.__window = cv.namedWindow(self.__stream_name)

    def end_stream(self):
        while True:
            try:
                im , self.t , self.p2_time = self.stream_queue.get_nowait()
                self.display_stream_img(im)
            except queue.Empty:
                self.stop_video_stream()
                break
        
    def stop_video_stream(self):
        while self.stream_queue.qsize() > 0:
            self.update_stream()

        cv.destroyWindow(self.__stream_name)
        
    def start_projector(self):
        self.projector.start()

    def stop_projector(self):
        self.projector.stop()
        self.__savefile.close()


    def update_projector(self):
        try:
            cmd , trigger_dict = self.projector_queue.get_nowait()
        except queue.Empty:
            self.projector.show()
            trigger = self.check_triggers()

            if trigger:
                stim = trigger.get_stimuli()
                trigger_dict = trigger.get_trigger_dict()

                if not stim:
                    stim = self.__get_stimuli(trigger_dict)

                self.projector.show_stream_and_stimuli(stim,trigger_dict['intensity'])
                    

                
        else:
            if cmd == 'Present':
                stim , intensity = self.__get_stimuli(trigger_dict)
                #print("Showing stimuli")
                #print(trigger_dict)
                self.show_stream_and_stimuli(stim,intensity)
                self.__record_stimuli(trigger_dict) #[SIC]
            elif cmd == 'Add':
                trigger_dict , stim = self.__prepare_stimuli(trigger_dict)
                print("Adding stimuli")
                #print(trigger_dict)
                self.projector.add_trigger(trigger_dict,stim)
                self.show_triggers()
            elif cmd == 'Calibrate':
                self.__init_kalman()
            elif cmd == 'Remove':
                self.show_triggers()
                self.projector.remove_trigger(trigger_dict)
                self.show_triggers()
            elif cmd == 'Background':
                self.projector.set_bgr_col(trigger_dict)
                self.bgr_col = trigger_dict

    def update_stream(self):
        try:
            im , self.t , self.p2_time = self.stream_queue.get_nowait()
            self.__process_img_func(im)
        except queue.Empty:
            pass

    def show_triggers(self,k=10):
        frame = self.projector.get_triggers_perimiter()

        for _ in range(k):
            self.update_stream()
            self.projector.show(frame)



    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    #*#*#*#*#*#*# ONLINE TRACKING #*#*#*#*#*#*#
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

    def init_online_processing(self):
        print("Initializing online processing...")
        dlc_proc = Processor()
        self.dlc_live = DLCLive(self.__config_path, processor=dlc_proc)        
        try:
            self.dlc_live.init_inference()
            
            qval = self.stream_queue.get()

            im = qval[0]

            self.__im_size = im.shape

            _ = self.dlc_live.get_pose(im)

        except tf.errors.ResourceExhaustedError as e:
            print("MUST KILL PYTHON SCRIPTS TO FREE UP GPU MEMORY")
            raise e

    def __set_threshold(self,new_threshold):
        if 0 <= new_threshold <= 1:
            self.__threshold = new_threshold

    def __dlc_to_unit_coords(self,pos):
        return np.array([2,-2])*(pos / self.__im_size) + np.array([-1,1]) 

    def online_pose(self,im):
        # Ouput data is 4x3 (4x2 coords , 4x1 likelihoods)
        self.__dlc_data = self.dlc_live.get_pose(im)

        if self.__validate_probs(self.__dlc_data):
            self.__coords = self.__dlc_data[:,:-1]
            self.__filtered_coords = self.__clean_coords_tmp(self.__coords) # These coords are returned as 2x4
            
            self.__current_snout_pos = np.array(self.__validate_coords(self.__filtered_coords[0,0],self.__filtered_coords[1,0]))
            self.__current_snout_pos_unit = self.__dlc_to_unit_coords(self.__current_snout_pos)
            self.__current_snout_proj_coords = self.projector._translate_coords(self.__current_snout_pos_unit)

            self.__current_midpt , self.__current_gaze_vector , self.__current_gaze_angle = self.__get_head_pose(self.__filtered_coords)
            self.__current_midpt = np.array(self.__validate_coords(self.__current_midpt[0],self.__current_midpt[1]))

            self.__current_midpt_unit = self.__dlc_to_unit_coords(self.__current_midpt)
            self.__current_midpt_proj_coords = self.projector._translate_coords(self.__current_midpt_unit)

            return True
        else:
            return False

    def process_stream_img(self,im):
        success = self.online_pose(im)

        if success:

            for [x,y] in self.__coords:
                rr , cc = draw.disk((y,x),radius=3,shape=im.shape)
                im[rr,cc] = 0

            
            for [x,y] in self.__filtered_coords.T:
                rr , cc = draw.disk((y,x),radius=3,shape=im.shape)
                im[rr,cc] = 0

            #print(f"Current midpt: {self.__current_midpt}")
            #print(f"Add current gaze vec: {self.__current_gaze_vector}*5")
            #print(f"Giving: {self.__current_midpt + self.__current_gaze_vector*5}")

            xmin , ymin = self.__validate_coords(   self.__current_midpt[0],
                                                    self.__current_midpt[1])
            xmax , ymax = self.__validate_coords(   self.__current_midpt[0] + self.__current_gaze_vector[0]*5,
                                                    self.__current_midpt[1] + self.__current_gaze_vector[1]*5)

            angle = self.__current_gaze_angle
            r = 60
            
            #x_add , y_add = self.__current_snout_pos + np.array([np.sin(angle) , np.cos(angle)])*r
            #x_minus , y_minus = self.__current_snout_pos - np.array([np.sin(angle) , np.cos(angle)])*r
            
            #xmax_a , ymax_a = self.__validate_coords(x_add,y_add)
            #xmax_m , ymax_m = self.__validate_coords(x_minus,y_minus)

            #rr , cc , val = draw.line_aa(int(self.__current_snout_pos[0]),int(self.__current_snout_pos[1]),int(xmax_a),int(ymax_a))
            #im[cc,rr] = val
            
            #rr , cc , val = draw.line_aa(int(self.__current_snout_pos[0]),int(self.__current_snout_pos[1]),int(xmax_m),int(ymax_m))
            #im[cc,rr] = val
            
            rr , cc , val = draw.line_aa(int(xmin),int(ymin),int(xmax),int(ymax))
            im[cc,rr] = val

        self.display_stream_img(im)
    
    def check_triggers(self):
        triggered = self.projector.check_triggers(self.__current_snout_proj_coords)

        if triggered:
            stim = triggered.get_stimuli()
            trigger_dict = triggered.trigger_dict
            
            if not stim:
                stim , intensity = self.__get_stimuli(trigger_dict)
            else:
                intensity = trigger_dict['intensity']

            self.show_stream_and_stimuli(stim,intensity)
            self.__record_stimuli(trigger_dict) #[SIC]

    

    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    #*#*#*#*#*#*# STREAM FUNCTIONS *#*#*#*#*#*#
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

    def show_stream_and_stimuli(self,stim,intensity):

        #t_show = 0
        #t_stream = 0
        save_time = True

        for stim_val in stim:
            #t_start = time.time()
            self.update_stream()
            
            if save_time:
                self.p1_time = time.time()
                self.p2_time_first = self.p2_time
                self.t_first = self.t

                self.projector.show_sequence([stim_val],1,intensity)

                self.p1_time = (self.p1_time + time.time()) / 2

                save_time = False
            else:
                self.projector.show_sequence([stim_val],1,intensity)

            
            #t_end = time.time()
            #t_end_stream = time.time()

            #t_show += t_start - t_end
            #t_stream += t_end - t_end_stream
            
        #print(f"Avg time to show sequence of length {len(stim)}: {(t_show / len(stim))*1000} ms")
        #print(f"Avg time to update steram: {(t_stream / len(stim))*1000 / len(stim)} ms")


    def display_stream_img(self,img,ms=1):
        cv.imshow(self.__stream_name,img)
        cv.waitKey(ms) # Will actually take around 10 ms to show one frame

    def close_video_stream(self):
        cv.destroyWindow(self.__stream_name)

    def create_savefile(self,filename):
        """
        Saves all stimuli actions to a csv
        Indexed with most recently showed camera frame
        """
        
        self.__savepath = filename

        self.__savefile = open(self.__savepath,'w',encoding='utf-8')
        self.__writer = csv.writer(self.__savefile)
        self.__writer.writerows([[  "Est time",
                                    "Interval min",
                                    "Interval max",
                                    "stim_type", 
                                    "size", 
                                    "start_pos", 
                                    "end_pos", 
                                    "speed" , 
                                    "intensity",
                                    "relative_to_mouse",
                                    "distance",
                                    "distance_unit",
                                    "angle_intervals",
                                    "random",
                                    "trigger_pos",
                                    "trigger_pos_unit",
                                    "trigger_rad1",
                                    "trigger_rad2",
                                    "trigger_rad1_unit",
                                    "trigger_rad2_unit",
                                    "trigger_index"]])


    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    #*#*#*#*#*#*# PRIVATE METHODS #*#*#*#*#*#*#
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

    #*#*#*#*#*#*# DLC LIVE METHODS *#*#*#*#*#*#

    def __validate_coords(self,x,y):
        
        xmax , ymax = self.__im_size

        if x < 0:
            x = 0
        elif x >= xmax:
            x = xmax - 1

        if y < 0:
            y = 0
        elif y >= ymax:
            y = ymax - 1

        return x , y

    def __validate_probs(self,arr,threshold=0.9):
        return all(arr[:,-1] >= self.__threshold)

    def __clean_coords(self,arr):
        # Kalman filter functions take 2x4 dim input
        try:
            cleaned_coords = predict_and_update(arr.T,self.__kalman_filters)
        except AttributeError:
            return arr.T

        return cleaned_coords


    def __get_head_pose(self,arr):
        # Pose functions take 4x2 dim input...

        arr_t = arr.T
        gzv = gaze_vec(arr_t)
        gaze_angle = h_angle_x(gzv)
        midpt = get_midpt(arr_t)

        return midpt , gzv , gaze_angle


    def __get_pose(self,arr):
        pose = get_output_variables_2d(arr)
        

    def __init_kalman(self,init_n=10,thresh=0.9):
        if not self.__online:
            self.init_online_processing()
        
        print("Calibrating Kalman Filter")

        n_max_tries = 500

        pbar = tqdm.tqdm(total=init_n,leave=False)

        i = 0
        starting_coords = np.zeros((2,4))
        tries = 0
        while((i < init_n) & (tries < n_max_tries)):
            tries += 1
            qval = self.stream_queue.get()

            im = qval[0]

            dlc_data = self.dlc_live.get_pose(im)

            if len(dlc_data) == 4:
                likelihoods = dlc_data[:,-1]
                p_ok = all(likelihoods >= thresh)

                if p_ok:
                    coords = dlc_data[:,:-1].T # Only extract relevant values (not probabilities)

                    starting_coords+=coords 

                    pbar.update(1)
                    i+=1

        if tries >= n_max_tries:
            print("Calibration failed, please try again.")
        else:
            starting_coords /= init_n

            self.__kalman_filters = create_kalman_filters(starting_coords,10)
            print("Kalman filter initialized using:")
            print(starting_coords)

            
            for x,y in starting_coords.astype(int).T:
                rr , cc = draw.disk((y,x),radius=5,shape=im.shape)

                im[rr,cc] = 1

            self.display_stream_img(im,1500)

        # After kalman filter has been calibrated we can do online tracking
        if self.__online:
            self.__process_img_func = lambda im : self.process_stream_img(im)
        else:
            self.__process_img_func = lambda im : self.display_stream_img(im)

        self.__clean_coords_tmp = lambda im : self.__clean_coords(im)


    def __get_relative_positions(self,args):

        r = args['distance_unit']
        angles = args['angle_intervals']
        
        print("Getting relative positions")

        if len(angles) == 2:
            angle_rads = np.random.uniform(angles[0],angles[1])
        elif len(angles) == 4:
            side = round(np.random.uniform())
            idx = side*2
            angle_rads = np.random.uniform(angles[idx],angles[idx+1])
        else:
            angle_rads = 0
    
        #print(f"Sampled angle {angle_rads} radians")

        angle_rads += self.__current_gaze_angle

        #proj_rad = np.array([(self.projector.xmax - self.projector.xmin)/2. , (self.projector.ymax - self.projector.ymin)/2.])

        print(f"Adding current gaze angle: {self.__current_gaze_angle} radians")

        #print(f"Current midpoint is (unit coords): {self.__current_midpt_unit}")
        print(f"to which we add: (sin(angle_rads)={np.sin(angle_rads)} , cos(angle_rads)={np.cos(angle_rads)})")
        #print(f"times normalized radius {r/2}")

        x , y = self.__current_midpt_unit + np.array([np.sin(angle_rads) , -np.cos(angle_rads)])*r
        
        print(f"Giving unit coords ({x},{y})")
        #x_t , y_t = self.projector._translate_coords((x,y))
        #print(f"Out comes translated coords ({x_t},{y_t})")

        return x , y
        

    #*#*#*#*#*#*# STIMULI METHODS *#*#*#*#*#*#


    def __prepare_stimuli(self,args):
        print("Preparing stimuli")
        try:
            assert type(args) == dict

            key = args['stim_type'].lower()

            assert key in ['shadow','moving dot','fixed dot']

            if args['relative_to_mouse']:
                print("Relative to mouse!")
                args['start_pos_func'] = lambda stim_args : self.__get_relative_positions(stim_args)
                stim = []
            else:
                if args['random']:
                    args['start_pos'] = np.random.uniform(-1,1,2)
                    args['end_pos'] = np.random.uniform(-1,1,2)

                args['start_pos_func'] = lambda stim_args : stim_args['start_pos']
            
                if key == 'shadow':
                    stim = self.projector.create_looming_shadow(5*args['size'],args['start_pos_func'](args),5,1 + 200*(1-args['speed']/100.))
                elif key == 'moving dot':
                    stim = self.projector.create_moving_dot(args['size'],args['start_pos_func'](args), args['end_pos'],1 + 200*(1-args['speed']/100.))
                elif key == 'fixed dot':
                    stim = self.projector.create_fixed_dot(args['size'],args['start_pos_func'](args),1 + 200*(1-args['speed']/100.))

        except AssertionError:
            stim = []

        return args , stim


    def __get_stimuli(self,args):
        try:
            assert type(args) == dict

            key = args['stim_type'].lower()
            
            assert key in ['shadow','moving dot','fixed dot']
            
            intensity = args['intensity']

            if args['relative_to_mouse']:
                print("Present stimuli relative to mouse!")
                args['start_pos_func'] = lambda stim_args : self.__get_relative_positions(stim_args)
            else:
                if args['random']:
                    args['start_pos'] = np.random.uniform(-1,1,2)
                    args['end_pos'] = np.random.uniform(-1,1,2)

                args['start_pos_func'] = lambda stim_args : stim_args['start_pos']
            
            if key == 'shadow':
                stim = self.projector.create_looming_shadow(5*args['size'],args['start_pos_func'](args),5,1 + 200*(1-args['speed']/100.))
            elif key == 'moving dot':
                stim = self.projector.create_moving_dot(args['size'],args['start_pos_func'](args), args['end_pos'],1 + 200*(1-args['speed']/100.),args['random'])
            elif key == 'fixed dot':
                stim = self.projector.create_fixed_dot(args['size'],args['start_pos_func'](args),1 + 200*(1-args['speed']/100.))
            else:
                stim = []
        except AssertionError:
            stim = []
            intensity = 0

        return stim , intensity



    def __record_stimuli(self,args):
        stim_type = args['stim_type']
        
        elapsed_time = round((self.p1_time - self.p2_time_first)*1000)
        print(f"Elapsed time: {elapsed_time} ms")
        est_time = 2*self.t_first + elapsed_time + self.__empirical_mean
        firstrow = [est_time, est_time - self.__empirical_error, est_time + self.__empirical_error]
        addrow = [val for val in args.values()]

        self.__writer.writerow(firstrow + addrow)
