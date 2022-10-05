import datetime , time , os , logging
from parsers import typeParser
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

class DirTracker:
    """
    This is a class object which holds and manipulates the current paths and names to use for saving files.
    It also keeps track of the current mouse subject ID (by user input).
    """
    def __init__(self,dirs):
        self._saved_dirs = []
        self.__savedFiles = {}
        self.__runIdxes = {}

        self.__datestamp = datetime.datetime.fromtimestamp(time.time())

        self.__dirs = dirs
        
        self.__scriptName = self.__dirs['script_name']
        
        self.__date = self.__datestamp.strftime("%Y%m%d")
        
        self.__fileName = self.__datestamp.strftime(F"{self.__date}_{self.__scriptName}")

        self.__dirs['savedir'] = os.path.join(self.__dirs['savedir'],self.__date)
        
        self.__currMouseName = None

        self.__check_for_previous_runs()


    def __check_for_previous_runs(self):
        """
        Check if there has been previous experiments run today with the same mouse.
        If so, update paths accordingly (if previous highest run was 2 then the new will be labeled 3).
        This is a bit hardcoded - it makes the assumption that after removing the last two parts of the
        filename we get the name of the ran script with runnumber at the end.
        """
        Path(os.path.join(self.__dirs['savedir'])).mkdir(parents=True,exist_ok=True)

        # Show all folders (mouse-idx folders) in current savedir (date-folder)
        mouse_idx_folders = next(os.walk(self.__dirs['savedir']))[1]
        
        # Check all mousefolders
        for mousefolder in mouse_idx_folders:
            highest_run_no = 0

            folderpath = Path(os.path.join(self.__dirs['savedir'],mousefolder))

            subfolder = next(os.walk(folderpath))[1]

            # Check all runfolders for previous runs
            for runfolder in subfolder:

                split_foldername = runfolder.split('_')
                rejoined_foldername = '_'.join(split_foldername[:-2])

                if rejoined_foldername == self.__fileName:
                    try:
                        run_idx = int(split_foldername[-1])
                        # Find the highest run number
                        if run_idx > highest_run_no:
                            highest_run_no = run_idx
                    except ValueError:
                        pass
                    
            mouseIdx = split_foldername[-2]

            self.__runIdxes[mouseIdx] = highest_run_no


    def __validate_makedir(self,given_dir):
        Path(os.path.join(self.__dirs['savedir'],given_dir)).mkdir(parents=True, exist_ok=True)

    def __make_save_dirs(self):
        self.__currFolderName = os.path.join(self.__dirs['savedir'], "Mouse_" + str(self.__currMouseName), self.__totalFileName)
        self.__validate_makedir(self.__currFolderName)
        
        if self.__currFolderName not in self.__dirs['savedir']:
            self._saved_dirs.append(self.__currFolderName)

    def __setup_tk(self):
        self.__root = tk.Tk()
        self.__root.title("New Mouse ID")
        self.__root.geometry('250x100')
        self.__root.focus()
        self.__font = lambda size=12 : tk.font.Font(family='Helvetica',size=size)

        self.__mouse_var = tk.StringVar(self.__root)
        self.__lbl_mouse = tk.Label(self.__root,text="Choose ID of next mouse:",font=self.__font())
        self.__lbl_mouse.grid(row=0,column=0)

        self.__mouse_options = [k for k in self.__runIdxes.keys()]

        try:
            self.__mouse_var.set(self.__mouse_options[0])
        except IndexError:
            self.__mouse_var.set("None")
            pass
        
        self.__mouse_opts = tk.OptionMenu(self.__root, self.__mouse_var, *self.__mouse_options)
        self.__mouse_opts.config(width=10, textvariable=self.__mouse_var, font=self.__font())
        self.__mouse_opts.grid(row=1,column=0)

        new_mouse_btn = tk.Button(master=self.__root, text="Add ID", command=self.__add_mouse_popup)
        new_mouse_btn.grid(row=1,column=1)

        ok_btn = tk.Button(master=self.__root, text="OK", command=lambda:self.__check_value(self.__mouse_var.get(),True))
        ok_btn.grid(row=2,column=0)

        self.__root.mainloop()
        
    def __add_mouse_popup(self):
        self.__popup = tk.Toplevel(self.__root)
        self.__popup.focus()

        tk.Label(self.__popup, text="Please enter new mouse ID:",font=self.__font()).grid(row=0,column=0)

        self.__new_mouse_var = tk.StringVar()
        self.__mouse_textbox = tk.Entry(self.__popup, textvariable=self.__new_mouse_var, width=4,font=self.__font())
        self.__mouse_textbox.grid(row=0,column=1)

        add_btn = tk.Button(master=self.__popup, text="Add", command=lambda: self.__check_value(self.__mouse_textbox.get(),False))
        add_btn.grid(row=0,column=2)

        self.__popup.mainloop()

    def __query_mouse_ID(self):
        self.__setup_tk()
        return int(self.__mouse_var.get())

    def __check_value(self,val,do_quit):
        try:
            new_val = int(val)
        except ValueError:
            messagebox.showerror(title="Error", message="Mouse ID must be an integer value!")
            self.__mouse_textbox.delete('0','end')
            self.__popup.focus()
            pass
        else:
            if do_quit:
                self.__root.destroy()
            else:
                if new_val not in self.__mouse_options:                    
                    self.__mouse_options.append(new_val)
                    self.__mouse_opts["menu"].add_command(label=new_val, command=lambda value=new_val: self.__mouse_var.set(new_val))

                self.__mouse_var.set(new_val)                    
                self.__popup.destroy()

    def create_new_subfolder(self,new_mouse_ID):
        #new_mouse_ID = typeParser('\nPlease enter the name (numbers only) of the next mouse: ', int)
        
        if new_mouse_ID in self.__runIdxes.keys():
            if new_mouse_ID != self.__currMouseName:
                self.__currMouseName = new_mouse_ID

            self.__runIdxes[self.__currMouseName] += 1
        else:
            self.__currMouseName = new_mouse_ID
            self.__runIdxes[self.__currMouseName] = 1

        logging.debug(F"Mouse {self.__currMouseName}\tRun {self.__runIdxes[self.__currMouseName]}\n")

        self.__mouseIndexedFileName = self.__fileName + F"_{self.__currMouseName}"
        self.__totalFileName = self.__mouseIndexedFileName + F"_{self.__runIdxes[self.__currMouseName]}"

        self.__make_save_dirs()

    def get_all_subfolders(self):
        """
        Returns list with paths to all the created directories this run.
        """
        return self._saved_dirs

    def get_filename(self):
        """
        Returns filename without any extensions.
        E.g. '20200811_behaviour2020_v_8_1'
        """
        return self.__totalFileName

    def get_dirs(self,key=None):
        """
        Returns dictionary containing current paths to scripts, savedir and arduino executable.
        Keys : "scripts", "ard_exec", "savedir"
        """
        if key is None:
            return self.__dirs
        elif key in self.__dirs.keys():
            return self.__dirs[key]

    def get_current_savepath(self,append_str='',save_format='',append=True):
        """
        Called from various threads to get new updated file path. When called upon
        it saves the requested string as this will be a created file by some thread.
        This is to allow for adding timestamps later (and more).

        The dict has file format endings as keys and dicts as values. The inner dicts have
        file name endings as keys and paths as values.
        E.g. savedFiles['csv']['timestamps'] = /absolute/path/to/20200811_behaviour2020_v_8_1_timestamps.csv
        and all other files ending with 'timestamps' of format 'csv' produced during the current run.
        """

        if save_format == '' and append_str == '':
            return os.path.join(self.__currFolderName, self.__totalFileName)

        if save_format[0] == '.':
            ending = save_format[1:]
        else:
            ending = save_format
        
        savepath = os.path.join(self.__currFolderName, self.__totalFileName) + '_' + append_str + '.' + ending
        
        if append:
            if ending in self.__savedFiles.keys():
                if append_str in self.__savedFiles[ending].keys():
                    if savepath not in self.__savedFiles[ending][append_str]:
                        self.__savedFiles[ending][append_str].append(savepath)
                else:
                    self.__savedFiles[ending][append_str] = [savepath]
            else:
                self.__savedFiles[ending] = {append_str:[savepath]}

        return savepath

    def get_all_saved_filenames(self):
        """
        Returns dictionary created in get_current_savepath during run.
        """
        return self.__savedFiles

    def get_mouse_dict(self):
        """
        Returns dict with ID of all mice that have run today as keys and their respective highest run index as value.
        """
        return self.__runIdxes

    def get_curr_mouse_ID(self):
        """
        Returns string with current mouse ID
        """
        return self.__currMouseName

if __name__ == "__main__":
    from optionHandler import OptionHandler

    program_constants = {   'fourcc':'123',
                            'video_format':'avi',
                            'rec_string':'recording',
                            'timestamp_format':'csv',
                            'timestamp_string':'timestamps',
                            'channels':1,
                            'cam_fps':550.,
                            'vid_fps':65.}

    # Ask user which computer they are on and adjust directories accordingly
    #dirs = dirParser()
    #scriptName = fileParser(dirs['scripts'])
    
    # Get user input for various parameter settings and choices
    optHandler = OptionHandler(program_constants)
    
    # Ask which script to run and create appropriate folders
    dirTracker = DirTracker(optHandler.get_dir_dict())
    dirTracker.create_new_subfolder(1)
    
    print(dirTracker.get_current_savepath())