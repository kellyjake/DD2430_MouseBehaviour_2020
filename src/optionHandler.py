from parsers import choiceParser , readCommandOptions , typeParser
from listPorts import serial_ports
import os 
from verbose_print import verboseprint
import tkinter as tk
from TkUIOptions import TkUIOptions

class OptionHandler:
    """
    Class for keeping track of user options. This might be changed later to a TkInter interface.
    """
    def __init__(self, program_constants):
        
        self.__tkUIOpt = TkUIOptions()
        self.__choice_dict = self.__tkUIOpt.get_choice_dict()
        self.__spec_dict = self.__tkUIOpt.get_spec_dict()
        self.__dir_dict = self.__tkUIOpt.get_dir_dict()

        # Input options read from header of chosen script.
        self.__optDict = readCommandOptions(self.__dir_dict['full_script_path'])
        self.__optDict['R'] = '[Start new run]'
        self.__optDict['Q'] = '[Save and exit]'
        
        self.__set_camera_specs(program_constants)

        self.__set_vprint()

    def __set_camera_specs(self,program_constants):
        for key,value in program_constants.items():
            self.__spec_dict[key] = value
        
    def __set_vprint(self):
        """
        Verbose print used for debug printing. Defined once to avoid if statements.
        """
        self.__vprint = verboseprint(self.__choice_dict['verbose'])


    def get_dict_value(self,key):
        try:
            val = self.__choice_dict[key]
        except KeyError:
            self.__vprint(F"Key {key} not recognized, returning None")
            return None
        self.__vprint(F"Returning {val} (key: {key})")
        return val


    def get_opt_dict(self):
        return self.__optDict


    def set_camera_dims(self, height, width):
        self.__spec_dict['height'] = height
        self.__spec_dict['width'] = width


    def get_vprint(self):
        return self.__vprint

    def get_camera_specs(self):
        return self.__spec_dict

    def get_dir_dict(self):
        return self.__dir_dict

    def kill(self):
        self.__tkUIOpt.hard_kill()
        self.__tkUIOpt = None