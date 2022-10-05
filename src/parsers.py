import os , sys , logging , string , datetime

def choiceParser(choices,query=''):
    """
    Presents options in choices, which may be a list or dict.
    If list, then it will be converted to a dict with the corresponding first
    letters of the alphabet as keys. Else keys of given dict are keys.

    Both key and value is returned as choice.

    A query may be given that is prompted to the user ( recommended :) )

    Returns none if no choices is empty.
    """
    print(F"\n{query}\n")

    if len(choices) == 0:
        print("No choices available, returning None!")
        return None , None
    
    if type(choices) == list:
        choiceDict = {i:k for i,k in zip(string.ascii_lowercase,choices)}        
    elif type(choices) == dict:
        choiceDict = choices
    else:
        print("Choices must be comprised in list or dictionary!")
        print(F"Not {type(choices)}")
        raise TypeError
    
    print("-- Please choose an option: --")
    print("------------------------------\n")
    for k,v in sorted(choiceDict.items()):
        print("{0}: {1}".format(k,v))
    print("------------------------------\n")

    choice = None
    
    while choice not in choiceDict.keys():
        choice = input(">> ")
        
    return choice , choiceDict[choice]
    


def readFileOptions(f,start_str,stop_str,comment_format,delim):
    """
    Attains user input options and descriptions from a given file and returns these in a dictionary. 
    The script must be formatted with start_str indicating when to start reading lines and extracting variables,
    stop_str indicates when to stop, comment_format indicates which characters stand before comments (// in C++, # in Python for example),
    and delim indicates what separates the option key from the description. 
    
    EXAMPLE:
    start_str = 'BEGIN_OPTION_SETUP'
    end_str = 'END_OPTION_SETUP'
    comment_format = '//'
    delim = '--'

    In script:

    other code
    other code

    //--BEGIN_OPTION_SETUP--
    //a--Option 1
    //b--Option 2
    //r--Option 3
    //--END_OPTION_SETUP--

    other code
    other code
    """
    optionDict = {}
    print(F"Trying to open {f}")

    f = open(f,'r')
    lines = f.readlines()
    addOpt = False

    for line in lines:
        words = list(filter(None,line.replace(comment_format,"").strip().split(delim))) # Weird looking line to get rid of empty strings.
       
        if len(words) == 0: continue

        if words[0] == stop_str:
            break

        if addOpt:
            try:
                optionDict[words[0]] = words[1]
            except IndexError:
                pass

        if words[0] == start_str:
            print(F"Got start_str: {start_str}")
            addOpt = True
        
    if not optionDict:
        print("-- OPTION SETUP MISSING --")
        print("Please add Keyboard Options at the top of chosen .ino script.")
        print("See README for more information.")
        raise IOError

    return optionDict


def readCommandOptions(f):
    """
    Specific use of readFileOptions for our formatting of our .ino files.
    """
    cmdDict = readFileOptions(f,'BEGIN_OPTION_SETUP','END_OPTION_SETUP','//','--')
    return cmdDict


def fileParser(scriptDir):
    """
    Presents all scripts located in script folder and parses user which one to run.
    """
    os.chdir(scriptDir)
    
    availableFiles = next(os.walk('.'))[1]
    fileDict = {i:f for i,f in zip(string.ascii_lowercase,availableFiles)}

    _ , fileName = choiceParser(fileDict,"Which routine would you like to run?")

    return fileName


def dirParser():
    """
    Deprecated since implementing UI with TkInter

    Parses the user which computer they are operating from.
    Important in order to get paths correct.
    Returns a dictionary with path to:
    scriptDir : Folder to .ino scripts
    ardDir : Folder to Arduino executable
    saveDir : Folder where to create subfolders for results
    """
    currComp , _ = choiceParser(["Lab Computer (Windows 10)","Office Computer (Ubuntu)","Magnus Home Laptop (Windows 10)","Other"],"Which computer are you operating from?")

    if currComp == 'a':
        # Dir where behavioural scripts exist
        scriptDir = r"C:\Users\user\Documents\Master_Program_Magnus\Ard_Scripts"
        
        # Dir where arduino exec files exist 
        ardDir = r"C:\Program Files (x86)\Arduino"
        
        # Dir to save results
        saveDir = r"E:\ExperimentalResults"

    elif currComp == 'b':
        scriptDir = "/home/titan/KI2020/Code/Ard_Scripts"
        ardDir = "/snap/arduino/41"
        saveDir = "/home/titan/KI2020/ExperimentalResults"

    elif currComp == 'c':
        scriptDir = r"C:\\Users\\magnu\\OneDrive\\Dokument\\KI\\KI2020\\Ard_Scripts"
        ardDir = r"C:\Program Files (x86)\Arduino"
        saveDir = r"C:\Users\magnu\OneDrive\Dokument\KI\ExperimentalResults"
    else:
        scriptDir = input("Please enter the path to the directory of the Arduino .ino scripts:\n")
        ardDir = input("Please enter the path to the directory of t he Arduino executable:\n")
        saveDir = input("Please enter directory where to save files:\n")

    return {"scripts": scriptDir , "ard_exec": ardDir , "savedir": saveDir}


def typeParser(parse_str,data_type):
    """
    Parse the user for a value that must be of type data_type.
    """
    while True:
        choice = input(parse_str)

        try:
            int_choice = data_type(choice)
        except ValueError:
            print(F"Choice must be of type {data_type.__name__}.")
            pass
        else:
            return int_choice


if __name__ == '__main__':
    dirs = dirParser()
    f = fileParser(dirs['scripts'])

    cmdDict = readFileOptions(os.path.join(dirs['scripts'],f,f + '.ino'),'BEGIN_OPTION_SETUP','END_OPTION_SETUP','//','--')
    
    print(cmdDict)