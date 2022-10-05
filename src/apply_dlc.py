import deeplabcut as dlc
import os , pickle , argparse

def extract_coords(config_path,videos):
    # Apply model and get csv file
    
    if type(videos) is list:
        video_list = videos
    elif type(videos) is str:
        filename, file_extension = os.path.splitext(videos)
        print(file_extension)
        if file_extension == '.p':
            f = open(videos,'rb')
            filedict , vid_specs = pickle.load(f)

            video_list = filedict[vid_specs['video_format']][vid_specs['rec_string']]
        else:
            video_list = [videos]

    model = dlc.analyze_videos(config_path, video_list, save_as_csv=True)

    # Create vid with labels
    dlc.create_labeled_video(config_path, video_list, trailpoints=1, draw_skeleton=True, videotype='.avi')
        
    newvids = []
    for vid in video_list:
        
        vidname , ext = os.path.splitext(vid)
        newname = vidname + model + ext 
        newvids.append(newname)

    print(F"New videos created:")
    for vid in newvids:
        print(vid)

def main(config_path,videos):
    extract_coords(config_path,videos)

if __name__ == '__main__':
    
    # defined command line options
    # this also generates --help and error handling
    CLI=argparse.ArgumentParser()

    CLI.add_argument(
        "--config_path",
        type=str,
        default='/home/titan/KI2020/DLC/KI2020_Project-Magnus-2020-08-28/config.yaml'
    )

    CLI.add_argument(
        "--video_list",
        type=str,
        default='/home/titan/KI2020/ExperimentalResults/20200911/Mouse_0411_10intensity_10on_15off_20/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1_recording.avi'
    )

    # parse the command line
    args = CLI.parse_args()
    # access CLI options

    config_path = args.config_path
    video_filename = args.video_list

    main(config_path,video_filename)