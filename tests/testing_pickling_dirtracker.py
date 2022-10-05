from parsers import fileParser , dirParser
from optionHandler import OptionHandler
from directoryTracker import DirTracker
import pickle
import cv2

program_constants = {   'fourcc':cv2.VideoWriter_fourcc(*'XVID'),
                            'video_format':'avi',
                            'rec_string':'recording',
                            'timestamp_format':'csv',
                            'timestamp_string':'timestamps',
                            'height':720,
                            'width':540,
                            'channels':1,
                            'cam_fps':550.,
                            'vid_fps':65.}

# Ask user which computer they are on and adjust directories accordingly
dirs = dirParser()
scriptName = fileParser(dirs['scripts'])

# Get user input for various parameter settings and choices
optHandler = OptionHandler(scriptName, program_constants)

# Ask which script to run and create appropriate folders
dirTracker = DirTracker(dirs,scriptName)

f = open('pickle_dir_test.p','wb')

pickle.dump([optHandler,dirTracker],f)

f.close()

g = open('pickle_dir_test.p','rb')

opt , dirt = pickle.load(g)

opt.get_camera_specs()
dirt.get_all_saved_filenames()