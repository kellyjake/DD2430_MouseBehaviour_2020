import pickle
import cv2
import os
import pickle_timestamps

def look_deeper(thisDir):
    subfolders = next(os.walk(thisDir))[1]
    print(subfolders)
    if len(subfolders) == 0:
        print(F"Fixing files in {thisDir}")
        fix_files(thisDir)
    else:
        for subDir in subfolders:
            print(subDir)
            joined_path = os.path.join(thisDir,subDir)
            look_deeper(joined_path)

def fix_files(final_dir):
    print(F"In fix_files with {final_dir}")
    files = next(os.walk(final_dir))[2]

    for d in files:
        print(d)
        splitted = d.split('.')[0].split('_')
        print(splitted)
        if splitted[-1] == 'recording':
            print(F"Found recording, appending with {d}")
            rescued_tracker['_DirTracker__savedFiles']['avi']['recording'].append(os.path.join(final_dir,d))
        elif splitted[-1] == 'timestamps':
            print(F"Found timestamps, appending with {d}")
            rescued_tracker['_DirTracker__savedFiles']['csv']['timestamps'].append(os.path.join(final_dir,d))


rescued_tracker = { 'vid_format':'avi',
                    'cam_rec_string':'recording',
                    'cam_timestamp_string':'timestamps',
                    '_DirTracker__savedFiles':{'avi':{'recording':[]},
                                                'csv':{'timestamps':[]}
                                                },
                    'vid_fourcc': cv2.VideoWriter_fourcc(*'XVID'),
                    'vid_fps':65.,
                    'cam_width':720,
                    'cam_height':540}

par_dir = r'e:\ExperimentalResults\20200728'

look_deeper(par_dir)

rescued_tracker['_DirTracker__savedFiles']



tracker_file = os.path.join(par_dir,'20200728_rescued_tracker.p')


"""
f = open(tracker_file,'wb')

pickle.dump(rescued_tracker, f)

f.close()
"""

"""
f = open(tracker_file,'rb')

rescued_tracker = pickle.load(f)
"""