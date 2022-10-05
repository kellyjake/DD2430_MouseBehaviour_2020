import os
from dlc_analysis import analyze_video
import tqdm

def files_to_process(folders,config_path,exclude_folders):
    bad_folders = []
    mov_files = []
    for folder in folders:

        if not (os.path.isdir(folder)):
            print(f"Skipping incorrect path {folder}")
            continue

        fullpath , subfolders , _ = next(os.walk(folder))

        for mouse_folder in subfolders:
            newpath = os.path.join(fullpath,mouse_folder)


            fullpath2 , run_folders , _ = next(os.walk(newpath))


            for run_folder in run_folders:
                finalpath = os.path.join(fullpath2,run_folder)
                
                if finalpath in exclude_folders:
                    continue

                files = next(os.walk(finalpath))[2]

                try:
                    mov_idx = get_file_idx(files)
                    video_file = os.path.join(finalpath,files[mov_idx])

                    mov_files.append(video_file)

                except AssertionError as e:
                    #print(f'Bad path! {finalpath}')
                    #print(e)
                    bad_folders.append(finalpath)

    print(mov_files)
    return mov_files , bad_folders

def get_file_idx(folder_content):
    n_avi = 0
    already_processed = False

    for idx,f in enumerate(folder_content):
        n , ext = os.path.splitext(f)

        if (ext == '.avi'):
            n_avi += 1
            mov_idx = idx

        if (ext == '.mp4'):
            already_processed = True
    
    assert not(already_processed) , 'Folder already processed!'
    assert (n_avi == 1) , 'Folder has non-postprocessed .avi files (from crashed session)!'
    
    return mov_idx

def main(folders,config_path,exclude_folders):
    mov_files , bad_folders = files_to_process(folders,config_path,exclude_folders)

    startframe = None
    endframe = None
    two_d = True
    usegpu = True

    print(F"Beginning to process {len(mov_files)} video(s).")

    print("Skipping the following folders:")
    for bad_folder in bad_folders:
        print(bad_folder)

    analyze = lambda mov : analyze_video(config_path,mov,startframe,endframe,two_d,usegpu)
    pbar = tqdm.tqdm(total=len(mov_files))

    for mov in mov_files:
        analyze(mov)
        pbar.update(1)



if __name__ == '__main__':
    import tensorflow as tf
    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)

    folders = ['/home/titan/KI2020/ExperimentalResults/20201126','/home/titan/KI2020/ExperimentalResults/20201128']
    config_path = '/home/titan/KI2020/DLC/KI2020_Training-Valter-2020-09-19/config.yaml'
    exclude = []
    
    main(folders,config_path,exclude)

    
