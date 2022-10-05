#try:
import deeplabcut as dlc
"""
except ImportError:
    print("Couldnt import deeplabcut!")
    pass
"""
import numpy as np
import csv , os , pickle , cv2 , tqdm , time
try:
    import tkinter as tk
    from tkinter import messagebox , filedialog
except ImportError:
    pass
import pandas as pd
import argparse
from kalman_filter import create_kalman_filters , predict_and_update
from pose2d import get_output_variables_2d
from pathlib import Path

class PoseExtractor:
    """
    Class object built to extract features from an experimental video of mouse in arena
    """
    def __init__(self,video_filename,dlc_config_path,two_dim=True,use_gpu=True):

        self.__baseline = None
        self.__a0_list , self.__b0_list , self.__c0_list = [] , [] , []
        self.__cam_fps = 500
        self.__config_path = dlc_config_path
        self.__video_filename = video_filename
        
        if use_gpu:
            self.__gputouse = 0
        else:
            self.__gputouse = None

        # This header must be changed if we change the extracted features (what get_output_variables)
        if two_dim:
            self.__header = [   "frame", 
                                "ba_len",
                                "g_vec_x" , "g_vec_y",
                                "cog_x" , "cog_y",
                                "b_vec_x" , "b_vec_y",
                                "head_angle",
                                "body_angle",
                                "head_body_angle"]
        else:
            self.__header = [  "frame",
                "tail_x" , "tail_y" , "tail_z" ,
                "l_ear_x" , "l_ear_y" , "l_ear_z" ,
                "r_ear_x" , "r_ear_y" , "r_ear_z" ,
                "snout_x" , "snout_y" , "snout_z" ,
                "gaze_vec_x" , "gaze_vec_y" , "gaze_vec_z",
                "cog_x" , "cog_y" , "cog_z",
                "body_vec_x", "body_vec_y", "body_vec_z",
                "head_angle" , "body_angle" ,
                "pitch", "roll", "head_body_angle"]
        


        # Used when extracting binary file
        self.__datatype = np.dtype([(head,np.float64) for head in self.__header])

    def __query_baseline(self):
        """
        Creates popup with instructions and queries the user to give start and end time (min,sec) of baseline.
        """
        self.__root = tk.Tk()
        self.__root.title("Select baseline")
        self.__root.withdraw()

        messagebox.showinfo(title="INFO",message=F'Please inspect the video\n{self.__labeled_video_filename} and note which part of video should be used as baseline (start second and end second - a short 3-5 second clip is good).\n\nThe baseline is used to compute the pose profile of the mouse and should be a sequence when the mouse in running in a straight line over the center of the arena.\n\nPress OK when you are ready to provide the start and end values.')

        self.__root.geometry('250x120')
        self.__center_window(self.__root)
        self.__root.focus()

        self.__tstart_min_var = tk.IntVar(self.__root)
        self.__tstart_sec_var = tk.IntVar(self.__root)

        self.__tend_min_var = tk.IntVar(self.__root)
        self.__tend_sec_var = tk.IntVar(self.__root)

        lbl_time = tk.Label(self.__root,text="Choose start and end time\nof baseline from video:")
        lbl_time.grid(row=0,column=0)

        tstart_lab = tk.Label(self.__root, text="Start time:")
        tstart_lab.grid(row=1,column=0)

        tstart_min_lab = tk.Label(self.__root, text="Min:")
        tstart_min_lab.grid(row=1,column=1)
        tstart_min = tk.Entry(self.__root, textvariable=self.__tstart_min_var,width=2)
        tstart_min.grid(row=1,column=2)

        tstart_sec_lab = tk.Label(self.__root, text="Sec:")
        tstart_sec_lab.grid(row=1,column=3)
        tstart_sec = tk.Entry(self.__root, textvariable=self.__tstart_sec_var,width=2)
        tstart_sec.grid(row=1,column=4)

        tend_lab = tk.Label(self.__root, text="End time:")
        tend_lab.grid(row=2,column=0)
        tend_min_lab = tk.Label(self.__root, text="Min:")
        tend_min_lab.grid(row=2,column=1)
        tend_min = tk.Entry(self.__root, textvariable=self.__tend_min_var,width=2)
        tend_min.grid(row=2,column=2)

        tend_sec_lab = tk.Label(self.__root, text="Sec:")
        tend_sec_lab.grid(row=2,column=3)
        tend_sec = tk.Entry(self.__root, textvariable=self.__tend_sec_var,width=2)
        tend_sec.grid(row=2,column=4)

        ok_btn = tk.Button(master=self.__root, text="OK", command=self.__check_times)
        ok_btn.grid(row=3,column=0)

        self.__root.mainloop()

    def __get_frame(self,minutes,seconds):
        """
        Converts time in video to specific frame, given the fps of the video
        :par minutes: int > 0
        :par seconds: int > 0
        :out: int frame number
        """
        return (minutes*60 + seconds)*self.__fps

    def __validate_frame(self,frame_start,frame_end):
        """
        Checks that valid frames are chosen as start/end to baseline
        :param frame_start: int - start frame of baseline
        :param frame_end: int - end frame of baseline
        :out: bool - valid frame?
        """

        print(F"Frame_start : {frame_start}")
        print(F"Frame_end : {frame_end}")
        print(F"n_frames: {self.__n_frames}")


        return (frame_end <= self.__n_frames) & (0 <= frame_start < frame_end)

    def __check_times(self):
        """
        Computes frame number from given start/end time. If valid labels the query interface closes, else error is shown asking for new values.
        """
        self.__end_frame = self.__get_frame(self.__tend_min_var.get(),self.__tend_sec_var.get())
        self.__start_frame = self.__get_frame(self.__tstart_min_var.get(),self.__tstart_sec_var.get())

        print(f"n_frames: {self.__n_frames}")
        print(f"fps: {self.__fps}")
        end_time = self.__n_frames/self.__fps
        minutes = end_time // 60
        seconds = end_time % 60

        if self.__validate_frame(self.__start_frame,self.__end_frame):
            print("Start frame: ", self.__start_frame)
            print("End frame: ", self.__end_frame)
            self.__root.destroy()
        else:
            messagebox.showerror(title="INFO",message=F"Please choose an end time within the length of the video ({minutes} mins {seconds} secs) and try again.")


    def __center_window(self,toplevel):
        """
        Method to center the tk window with root toplevel
        """
        toplevel.withdraw()
        toplevel.update_idletasks()
        screen_width = toplevel.winfo_screenwidth()
        screen_height = toplevel.winfo_screenheight()

        x = int(screen_width/2 - toplevel.winfo_reqwidth()/2)
        y = int(screen_height/2 - toplevel.winfo_reqheight()/2)

        toplevel.geometry(F"+{x}+{y}")
        toplevel.deiconify()


    def set_frame_start_stop(self,start,end):
        """
        For manual setting of start/end frame. For debug to avoid query box.
        """
        self.__start_frame = start
        self.__end_frame = end


    def create_baseline(self,max_frames=2500,baseline_threshold=0.9,start_frame=None,end_frame=None):
        """
        Computes a "baseline" pose descriptor. Max value of pose features over a manually selected
        timeframe in video seen as representative for the mouse profile. Preferably time interval when mouse is running traight across the center of arena.
        Also preffered is to check that labels are correctly placed during this interval.

        TODO: This function needs a more robust measure of baseline.

        :param video_filename:      path to video to be analylzed
        :param dlc_config_path:     path to deeplabcut config file for model to be used to extract coords
        :param max_frames:          number of frames to average over when computing baseline
        :param baseline_threshold:  threshold for accepting label by DLC model when creating baseline
        :param start_frame:         see end_frame
        :param end_frame:           if start_frame and end_frame given and both being float or integers, use these as
                                    baseline start/end frame and dont open query box. Mainly for debug purposes.

        :out:                       list of length 3 with baseline [a0,b0,c0]:
                                        a0: tail to midpt at back of head,
                                        b0: ear to ear,
                                        c0: midpt back of head to nose
        """

        self.__max_frames = max_frames

        self.__create_videos()

        # If values given don't show querybox
        if (bool(start_frame) & bool(end_frame)):
            if (self.__validate_frame(start_frame,end_frame)):
                print(F"Got start and end frames: ({start_frame},{end_frame})")
                print(self.__validate_frame(start_frame,end_frame))
                self.set_frame_start_stop(start_frame,end_frame)
        else:
            self.__query_baseline()
        
        split = os.path.splitext(self.__video_filename)[0]
        
        bl_filename = split + '_baseline.npy'

        # If too large time interval is given, reduce to max_frames
        if (self.__end_frame - self.__start_frame) > self.__max_frames:
            self.__end_frame = self.__start_frame + self.__max_frames

        print("Creating baseline...")
        pbar = tqdm.tqdm(total=(self.__end_frame - self.__start_frame))
        starting_coords = np.zeros((2,4))

        with open(self.__data_dlc,'r') as f:
            reader=csv.reader(f)
            for _ in range(3): next(reader)
            #Take out the first 10 rows
            i  = 0
            for row in reader:
                try:
                    float_row = map(np.float64,row)

                    frame_no , snout_x , snout_y , snout_p , l_ear_x , l_ear_y , l_ear_p , r_ear_x , r_ear_y , r_ear_p , tail_x , tail_y , tail_p = float_row
                    coords = np.array([[snout_x,l_ear_x,r_ear_x,tail_x],
                                        [snout_y,l_ear_y,r_ear_y,tail_y]])
                    starting_coords+=coords
                    i+=1
                except (TypeError, ValueError):
                    continue
                if i == 10:
                    break
        
        kalman_filters = create_kalman_filters(starting_coords/10)


        with open(self.__data_dlc,'r') as f:
            reader=csv.reader(f)

            # Read and discard headers (always written by DLC - hopefully always 3 rows long...)
            for _ in range(3): next(reader)

            # Start from given frame
            for _ in range(self.__start_frame-1): next(reader)

            # Until endframe - extract poses
            for i in range(self.__end_frame - self.__start_frame):
                accepted = True
                row = next(reader)

                try:
                    _ , snout_x , snout_y , snout_p , l_ear_x , l_ear_y , l_ear_p , r_ear_x , r_ear_y , r_ear_p , tail_x , tail_y , tail_p = map(np.float64,row)
                except TypeError:
                    pass
                
                # Apply Kalman filter to smooth outlier labels
                filtered_coords = predict_and_update(np.array([   [snout_x,l_ear_x,r_ear_x,tail_x],
                                                                  [snout_y,l_ear_y,r_ear_y,tail_y]  ]), 
                                                                  kalman_filters)

                # Check so that DLC has made a confident prediction for all labels
                threshold_passed = all(p >= baseline_threshold for p in [snout_p, l_ear_p, r_ear_p, tail_p])

                if threshold_passed:
                    snout , l_ear , r_ear , tail = filtered_coords.T

                    head_midpt = (l_ear + r_ear)/2.
                                                                # Same terminology as in pose.py for coherence:
                    a0 = np.linalg.norm(head_midpt - tail,2)    # a0: tail to midpt at back of head,
                    b0 = np.linalg.norm(l_ear - r_ear,2)        # b0: ear to ear,
                    c0 = np.linalg.norm(head_midpt - snout,2)   # c0: midpt back of head to nose

                    self.__a0_list.append(a0)
                    self.__b0_list.append(b0)
                    self.__c0_list.append(c0)

                pbar.update(1)

        # Get max feature as baseline
        self.__avg_a0 , self.__avg_b0 , self.__avg_c0 = map(np.nanmax,[self.__a0_list,self.__b0_list,self.__c0_list])
        self.__baseline = np.array([self.__avg_a0, self.__avg_b0, self.__avg_c0])

        concat = np.array([self.__a0_list,self.__b0_list,self.__c0_list])

        # Save data to csv for eventual inspection
        concat.tofile(bl_filename)

        print("Baseline saved in ", bl_filename)

        return concat


    def get_baseline(self,return_all=False):
        """
        Returns computed baseline. Mainly for debugging.
        """
        if return_all:
            return self.__baseline , [self.__a0_list,self.__b0_list,self.__c0_list]
        else:
            return self.__baseline


    def get_header(self):
        """
        Returns header of csv file.
        """
        return self.__header


    def extract_features(self,baseline=None,threshold=0.9,save_as_csv=False):
        """
        Reads DLC csv file created in create_baseline and computes and saves all features specified in pose.py.

        :param baseline:    optional manual baseline: np.array dim (1 x 3) ((3,) is fine). if not given the baseline in created_baseline is used
        :param threshold:   probability threshold for label predictions by DLC. higher = better quality but less labels
        :param save_as_csv: if true also saves results into csv file: takes additional minute or so.

        :out:               path to created binary (and csv if save_as_csv) file
        """

        if baseline:
            if (type(baseline) == list) & (len(baseline) == 3):
                self.__baseline = np.array(baseline)
            elif (type(baseline) == type(np.array([])) & (len(baseline) == 3)):
                self.__baseline = baseline
            else:
                raise TypeError

        split_filename = self.__video_filename.split(os.path.sep)[:-1]
        times = []

        name = split_filename[-1] + '_pose_data.npy'
        split_filename.append(name)
        self.__savefile_binary = os.path.sep.join(split_filename)

        pbar = tqdm.tqdm(total=self.__n_frames)

        #Initialize the Kalman filter with the mean of the ten first coordinates
        starting_coords = np.zeros((2,4))
        with open(self.__data_dlc,'r') as f:
            reader=csv.reader(f)
            for _ in range(3): next(reader)
            #Take out the first 10 rows
            i  = 0
            for row in reader:
                try:
                    float_row = map(np.float64,row)

                    frame_no , snout_x , snout_y , snout_p , l_ear_x , l_ear_y , l_ear_p , r_ear_x , r_ear_y , r_ear_p , tail_x , tail_y , tail_p = float_row
                    coords = np.array([[snout_x,l_ear_x,r_ear_x,tail_x],
                                        [snout_y,l_ear_y,r_ear_y,tail_y]])
                    starting_coords+=coords
                    i+=1
                except (TypeError, ValueError):
                    continue
                if i == 10:
                    break
        
        kalman_filters = create_kalman_filters(starting_coords/10)


        print("Generating pose data...")
        with open(self.__data_dlc,'r') as f:
            with open(self.__savefile_binary,'wb') as savefile:

                datatype = [(head,np.float64) for head in self.__header]

                reader=csv.reader(f)

                # Read and discard headers
                for _ in range(3): next(reader)

                # Read all lines in DLC csv file and compute features
                for row in reader:
                    try:
                        float_row = map(np.float64,row)

                        frame_no , snout_x , snout_y , snout_p , l_ear_x , l_ear_y , l_ear_p , r_ear_x , r_ear_y , r_ear_p , tail_x , tail_y , tail_p = float_row

                        # Since the camera is operating at 500 Hz, each frame corresponds to 2 ms, but DLC gives
                        # the number of the frame in the order. Thus we must multiply by 1000(ms)/FPS = 2
                        frame_no=int(frame_no*1000/self.__cam_fps)
                    except (TypeError, ValueError):
                        continue

                    filtered_coords = predict_and_update(np.array([ [snout_x,l_ear_x,r_ear_x,tail_x],
                                                                    [snout_y,l_ear_y,r_ear_y,tail_y]  ]),
                                                        kalman_filters)

                    threshold_passed = all(p >= threshold for p in [snout_p, l_ear_p, r_ear_p, tail_p])

                    # Check so that DLC made a confident prediction about all labels
                    if threshold_passed:
                        snout , l_ear , r_ear , tail = filtered_coords.T

                        pose_descriptor = get_output_variables_2d([snout,l_ear,r_ear,tail])

                        extracted_values = np.asarray(self.__extract_all_2d(frame_no,pose_descriptor),dtype=np.float64)

                        # Saves data in binary form to save size and speed
                        # This means data cannot be opened and viewed as text,
                        # but must be read and loaded as binary first
                        extracted_values.tofile(savefile)

                    pbar.update(1)

        print(F"Done! Results saved in binary file {self.__savefile_binary}.")

        if save_as_csv:
            savefile_csv = self.save_binary_to_csv()
            return [self.__savefile_binary , savefile_csv]

        return self.__savefile_binary


    def save_binary_to_csv(self, filename_in=None):
        """
        Method for converting binary pose data to a csv file.
        Made for Dimitrios to be able to analyze data by himself essentially.

        :param filename_in: can give path to any binary data file and it will be converted
                            and saved in the same directory with header as specified in the
                            init method. If not given, uses binary file created in extract_features.

        :out:               path to created csv file
        """

        filename = filename_in if filename_in else self.__savefile_binary
        split = os.path.splitext(filename)[0]
        savefile_csv = split + '.csv'

        print(F"Loading binary data from {filename}")

        with open(filename,'rb') as f:
            df = pd.DataFrame.from_records(np.fromfile(f,dtype=self.__datatype),columns=self.__header)
            df.to_csv(savefile_csv,index=False)

            print(F"Done! Results saved in csv file {savefile_csv}")

        return savefile_csv


    def create_df(self,filename_in=None):
        """
        Creates a pandas dataframe from binary file or csv file.

        :param filename_in: if specified this file will be loaded as a dataset. otherwise the data created in extract_features is loaded
        """
        filename = filename_in if filename_in else self.__savefile_binary

        ending = filename.split('.')[-1]


        if ending == 'npy':
            print(F"Loading binary data from {filename}")
            df = pd.DataFrame.from_records(np.fromfile(filename,dtype=self.__datatype),columns=self.__header)
        elif ending == 'csv':
            print(F"Loading csv data from {filename}")
            df = pd.read_csv(filename_in)
        else:
            print(F"Filetype {ending} not supported! Dataframe must be created from csv or npy file!")
            raise TypeError

        return df

    def create_videos(self):
        print(self.__video_filename)

        cap = cv2.VideoCapture(filename=self.__video_filename,apiPreference=cv2.CAP_FFMPEG)
        self.__n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        print(f"Analysing video... {self.__video_filename}")
        # Apply model and get csv file
        self.__model = dlc.analyze_videos(self.__config_path, [self.__video_filename], save_as_csv=True,gputouse=self.__gputouse)
        print(f"Model name: {self.__model}")

        print(f"Creating labeled video...")
        # Create vid with labels
        dlc.create_labeled_video(self.__config_path, [self.__video_filename], trailpoints=1, draw_skeleton=True, videotype='.avi')
        print("Created labeled video!")

        # analyze_videos returns path returns the name of the model, which then must be appended to the video name
        # to get the path to the data file. (This is how DLC data files are formatted)
        split = os.path.splitext(self.__video_filename)[0]
        new_name = split + self.__model + '.csv'
        self.__labeled_video_filename = split + self.__model + '_labeled' + '.mp4'
        self.__data_dlc = new_name

        assert os.path.isfile(self.__labeled_video_filename)
        assert os.path.isfile(self.__data_dlc)

        print(f"Created video {self.__labeled_video_filename}")

    def __extract_all_2d(self,frame,arr):
        """
        Method to explicitly extract all values in all dimensions from pose array with id frame
        Used to save data into copmrehensive data files.

        Based on the number of features produced in pose.py

        :param frame:   frame number
        :param arr:     np array with dims ((4,3) , (3,) , (3,) , (3,) , (1,) , (1,) , (1,) , (1,) , (1,))

        :out:           list of length sum(arr.shape) + 1 with scalars
        """

        ba_len, [g_vec_x , g_vec_y], [cog_x , cog_y], [b_vec_x , b_vec_y] , head_angle, body_angle, head_body_angle = arr

        full_arr = [frame, ba_len,
                    g_vec_x , g_vec_y ,
                    cog_x , cog_y,
                    b_vec_x , b_vec_y,
                    head_angle , body_angle ,
                    head_body_angle]

        return full_arr

    def __extract_all(self,frame,arr):
        """
        Method to explicitly extract all values in all dimensions from pose array with id frame
        Used to save data into copmrehensive data files.

        Based on the number of features produced in pose.py

        :param frame:   frame number
        :param arr:     np array with dims ((4,3) , (3,) , (3,) , (3,) , (1,) , (1,) , (1,) , (1,) , (1,))

        :out:           list of length sum(arr.shape) + 1 with scalars
        """

        x_3d, g_vec, cog, b_vec, head_angle, body_angle, pitch, roll, head_body_angle = arr

        [tail_x , tail_y , tail_z] , [l_ear_x , l_ear_y , l_ear_z] , [r_ear_x , r_ear_y , r_ear_z] , [snout_x , snout_y , snout_z] = x_3d
        g_vec_x , g_vec_y , g_vec_z = g_vec
        cog_x , cog_y , cog_z = cog
        b_vec_x , b_vec_y , b_vec_z = b_vec

        full_arr = [frame,
                    tail_x , tail_y , tail_z ,
                    l_ear_x , l_ear_y , l_ear_z ,
                    r_ear_x , r_ear_y , r_ear_z ,
                    snout_x , snout_y , snout_z ,
                    g_vec_x , g_vec_y , g_vec_z ,
                    cog_x , cog_y , cog_z ,
                    b_vec_x , b_vec_y , b_vec_z ,
                    head_angle , body_angle ,
                    pitch , roll , head_body_angle]

        return full_arr


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def analyze_video(config_path,video_filename,startframe,endframe,two_d,use_gpu):

    pose = PoseExtractor(video_filename,config_path,two_d,use_gpu)

    if not two_d:
        pose.create_baseline(video_filename,config_path,start_frame=startframe,end_frame=endframe)
    else:
        pose.create_videos()

    savefile_binary = pose.extract_features()

    savefile_csv = pose.save_binary_to_csv()

def main(config_path,video_filename,startframe,endframe,two_d,use_gpu):

    assert os.path.isfile(config_path)
    assert os.path.isfile(video_filename)

    analyze_video(config_path,video_filename,startframe,endframe,two_d,use_gpu)

if __name__ == "__main__":

    # defined command line options
    # this also generates --help and error handling
    CLI=argparse.ArgumentParser()

    CLI.add_argument(
        "--framestart",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        default=None,  # default if nothing is provided
    )
    CLI.add_argument(
        "--frameend",
        type=int,  # any type/callable can be used here
        default=None,
    )
    #'/home/titan/KI2020/DLC/KI2020_Project-Magnus-2020-08-28/config.yaml' # - initial model
    CLI.add_argument(
        "--config_path",
        type=str,
        default='/home/titan/KI2020/DLC/KI2020_Training-Valter-2020-09-19/config.yaml'
    )

    CLI.add_argument(
        "--video",
        type=str
    )
    CLI.add_argument(
        "--two_dim",
        type=str2bool,
        default=True
    )
    CLI.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True
    )

    # parse the command line
    args = CLI.parse_args()
    # access CLI options

    config_path = args.config_path
    video_filename = args.video
    framestart = args.framestart
    frameend = args.frameend
    two_d = args.two_dim
    usegpu = args.use_gpu

    main(config_path,video_filename,framestart,frameend,two_d,usegpu)
