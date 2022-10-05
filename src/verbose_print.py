import logging , sys

"""
Allows for easy definition of a verbose print function
"""

def noprint(*a ,**k):
    return None
    
def verboseprint(verbose):
    return lambda msg : logging.debug(msg) if verbose else noprint

if __name__ == "__main__":
    
    # Create logfile to save errors
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler('test.log')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    vprint = verboseprint(True)

    logging.debug("hej")
    vprint("Hello there2")
    