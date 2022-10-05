import os , sys , argparse
#os.chdir('src')
from add_timestamps import add_timestamps_multiple_files
import pickle

def main(fileName):
    """
    Script for loading pickled tracker and opthandler from previous run to add timestamps.
    Example usage:
    python pickle_timestamps.py /home/titan/KI2020/ExperimentalResults/20200811/Mouse_7/20200811_behaviour2020_iv_7_2/20200811_behaviour2020_iv_7_2_tracker.p
    
    (The easiest way to get the absolute is to drag the file into the command prompt)
    """
    print(fileName)
    try:
        f = open(fileName,'rb')
        dirTracker , optHandler = pickle.load(f)
        added = add_timestamps_multiple_files(dirTracker,optHandler)
    except TypeError as e:
        print(e)
        print(added)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == "__main__":
    # defined command line options
    # this also generates --help and error handling
    CLI=argparse.ArgumentParser()
    
    CLI.add_argument(
        "--pickle",
        type=str
    )

    # parse the command line
    args = CLI.parse_args()
    # access CLI options

    pickle_path = args.pickle
    
    main(pickle_path)