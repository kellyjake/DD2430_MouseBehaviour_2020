import cv2 , imutils , os , faulthandler , p_tqdm , functools , time , argparse
from numpy import linspace

def add_timestamps_multiple_files(all_files, vid_specs, pos=['right','bottom'],pixShift=[0,0],col=(255,255,255)):
    """
    Adds timestamps to all video recordings with proper formatting that exist in the DirTracker directory (all videos produced from the same session),
    given that the timestamps exist. If timestamps are already added, then those videos are skipped.

    Utilizes multiprocessing to try and speed up the process, but it still takes quite some time. For 30-40 minute videos count with about 10-20 minutes per video.
    I did 10 videos between 20-40 minutes, and it took 1,5 hours (on the Lab PC).

    Aborted timestamp addings will have to be restarted from the start.
    """
    faulthandler.enable()

    assert (type(all_files) is dict) & (type(vid_specs) is dict)
    
    try:
        file_list = all_files[vid_specs['video_format']][vid_specs['rec_string']]
    except KeyError as e:
        print(F"Tried to do all_files[{vid_specs['video_format']}][{vid_specs['rec_string']}]")
        print(all_files)
        print(F"Guessing {vid_specs['rec_string']} not in keys above")
        print(e)
        raise KeyError

    try:
        timestamp_list = all_files[vid_specs['timestamp_format']][vid_specs['timestamp_string']]
    except KeyError as e:
        print(F"Tried to do all_files['csv'][{vid_specs['timestamp_string']}]")
        print(all_files)
        print(F"Guessing {vid_specs['timestamp_string']} not in keys above")
        print(e)
        raise KeyError

    nFiles = len(file_list)
    
    # Make sure every video has a timestamp file
    assert nFiles == len(timestamp_list)

    task_list = []
    
    for i in range(nFiles):
        task_list.append([file_list[i],timestamp_list[i]])

    print(F"Beginning to process {nFiles} video(s). \nThis may take a while (up to 10 mins per video depending on size).\nProgress bar will be slow, but have patience!\n(It will be 0% until the first video is processed.)")

    # This function gives us a nice progress bar and applies multiprocessing pooling on an unsorted list.
    # https://github.com/swansonk14/p_tqdm
    # Functools allows for add_timestamps_multiproc function to accept multiple arguments (vid_specs) as well as list of tasks.
    p_tqdm.p_umap(add_timestamps_multiproc, task_list)
    
    return True

def add_timestamps_multiproc(task,pos=['right','bottom'],pixShift=[0,0],col=(255,255,255),ret_filename=False):
    """
    Method called by each processor when processing specific video and timestamp file (task is a list [vid path,timestamps path])
    pos can be used to vary positioning of timestamp. Valid parameters are 'right', 'left' and 'bottom','top'. 
    Any other string will default to 'right' and 'bottom'.
    This positioning depends on the dimension of the image and may fall outside of the frame if using too small width and height.
    pixShift shifts the timestamps in 
    
    Increasing pixShift[0] causes timestamps to go further right and vice versa.
    Increasing pixShift[1] causes timestamps to go further up and vice versa.
    ... Although I am not 100 % certain. Probably messing with these will cause timestamps to partially go outside of the frame :)

    col indicates color of text in RGB coding. Defaults to white.
    """
    try:
        vidFileName , timestampFileName = task
    except ValueError:
        print("Task must be a list of videofilename and timestamp filename!")
        print(task)
        raise ValueError

    split_name = vidFileName.split('.')
    new_name = '.'.join(split_name[:-1]) + '_timed.' + split_name[-1]
    
    cap = cv2.VideoCapture(vidFileName)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if video with timestamps already exists
    if os.path.isfile(new_name):
        cap_timed = cv2.VideoCapture(new_name)
        
        n_frames_timed = int(cap_timed.get(cv2.CAP_PROP_FRAME_COUNT))

        # Are all timestamps already added?
        if n_frames_timed == n_frames:
            print(F"File already timestamped: {new_name}")
            doWork = False
        else:
            doWork  = True

        cap_timed.release()
    else:
        doWork = True
    
    if doWork:
        csv_file = open(timestampFileName,'r')
        
        all_values = list(filter(None,csv_file.read().split('\n'))) # Removing newlines

        try:
            freq , t_start , t_end = [int(val.split(',')[1]) for val in all_values]
        except (ValueError , TypeError , IndexError) as e:
            print(F"Wrong formatting in {timestampFileName}")
            print(all_values)
            print("Should have format:\n ['Freq,int', 't_start,int', 't_end,int']")
            print("Please change the erreonous value and try again")
            print(e)
            raise Exception
        
        timestamps = [t for t in linspace(t_start,t_end,n_frames)]

        vidWriter_timed = cv2.VideoWriter(new_name, fourcc, fps, (width, height), False)

        print(F"Timestamp filename: {timestampFileName}\nVideo filename: {vidFileName}\nFrames in video: {n_frames}")
        print(F"Number of actual timesteps: {round((t_end-t_start)/(1000/freq))}")
        print(F"Compressed into: {len(timestamps)} timestamps due to frame-loss")
        
        t_start = time.time()
        
        check_n_frames = 1000

        for i in range(check_n_frames):
            success , frame = cap.read()
            if success:
                try:
                    stamped_frame = add_time(frame, timestamps[i], color=col)
                except IndexError:
                    print(F"Ran out of timestamps at frame {i}")
                    print(F"(File {new_name})")
                    break

                vidWriter_timed.write(stamped_frame)

        t_stop = time.time()

        dur_per_frame = (t_stop - t_start)/check_n_frames

        est_time = round((n_frames - check_n_frames)*dur_per_frame/60.,1)

        print(F"Estimated processing time: {est_time} minutes\n")

        for i in range(check_n_frames,n_frames + 1):
            success , frame = cap.read()
            if success:
                try:
                    stamped_frame = add_time(frame, timestamps[i], color=col)
                except IndexError:
                    print(F"Ran out of timestamps at frame {i}")
                    print(F"(File {new_name})")
                    break

                vidWriter_timed.write(stamped_frame)
        
        cap.release()
        vidWriter_timed.release()

        print(F"Video with {n_frames} frames took {round((time.time() - t_start)/60.,1)}")

        if ret_filename:
            return new_name


def add_time(img,timestamp,pos=['right','bottom'],pixShift=[0,0], monoChrom=True, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(255,255,255), show=False):
    """
    Places timestamp on image at pos. Timestamp must be string castable to int.
    Other fonts, scales and thicknesses can be used, but the defaults work the best for our current video dimensions.
    This function will be called hundred of thousand of times when called from add_timestamps_multiproc so I recommend to use show=False :)
    """
    height , width , _ = img.shape

    yPos = 15 if pos[0] == 'left' else width - 12
    xPos = height - 220 if pos[1] == 'top' else 5

    """
    if pos[0] == 'left':
        yPos = 15
    else:
        yPos = width - 12

    if pos[1] == 'top':
        xPos = height - 220
    else:
        xPos = 5
    """

    org = (xPos + pixShift[1], yPos + pixShift[0])
    
    # Must rotate image because cv2.putText only puts it horizontally (we want vertically)
    rotIm = imutils.rotate_bound(img, 90)
    try:
        int_timestamp = round(timestamp,2)
    except ValueError:
        print("Got bad timestamp ", timestamp)
        return img

    printStr = F"{int_timestamp}"
    
    img_t = cv2.putText(rotIm, printStr, org, fontFace, fontScale, color, thickness, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

    if monoChrom:
        # Keep only one channel
        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
    
    # Rotate back
    unRot = imutils.rotate_bound(img_t, -90)
    
    if show:
        cv2.imshow("img",unRot)
        cv2.waitKey()
    
    return unRot

def main(videos,timestamps,pos,pixshift,textcol):
    name = add_timestamps_multiproc([videos[0],timestamps[0]],pos,pixshift,textcol,True)
    
    print(F"Created video with timestamps: {name}")

if __name__ == "__main__":
    """
    all_files, timestamps, pos=['right','bottom'],pixShift=[0,0],col=(255,255,255)
    """

    # defined command line options
    # this also generates --help and error handling
    CLI=argparse.ArgumentParser()
    
    CLI.add_argument(
        "--video",
        type=str,
        nargs=1
    )
    CLI.add_argument(
        "--timestamps",
        type=str,  # any type/callable can be used here
        nargs=1
    )
    CLI.add_argument(
        "--textpos",
        type=str,
        nargs=2,
        default=['right','bottom']
    )
    CLI.add_argument(
        "--pixshift",
        type=int,
        nargs=2,
        default=[0,0]
    )
    CLI.add_argument(
        "--textcol",
        type=int,
        nargs=3,
        default=[255,255,255]
    )
    # parse the command line
    args = CLI.parse_args()
    # access CLI options

    video = args.video
    timestamps = args.timestamps
    pixshift = args.pixshift
    pos = args.textpos
    textcol = args.textcol

    main(video,timestamps,pos,pixshift,textcol)