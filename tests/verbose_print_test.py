import logging , sys

"""
Allows for easy definition of a verbose print function
"""

class VerbosePrint:

    def __init__(self, filename):
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        rootLogger = logging.getLogger()
        rootLogger.setLevel(logging.DEBUG)

        fileHandler = logging.FileHandler(filename)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

    def __noprint(self, *a ,**k):
        return None
        
    def get_vprint(self, verbose=True):
        return lambda msg : logging.debug(msg) if verbose else self.__noprint
    
if __name__ == "__main__":
    
    verbose = VerbosePrint('test.log')
    vprint = verbose.get_vprint(True)

    vprint("Hello there2")
    