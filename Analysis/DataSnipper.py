import numpy as np
import csv , os , argparse , pickle
import pandas as pd


class DataSnipper():
    """
    Author: Magnus Pierrau
    Date: 2020.11.17

    DataSnipper allows for extraction of random or specific rows from a CSV file.
    """

    def __init__(self,filename,n_header_lines=1,verbose=False):
        """
        Initialize datasnipper

        :par filename:  full path to CSV file to be processed
        :par n_header:  number of lines with header text in CSV
        :par verbose:   print explicit output
        """

        self.verbose = verbose
        self.__n_header_lines = n_header_lines

        if not os.path.isfile(filename):
            print(f"{filename} is not a valid path!")
        else:
            self.__filename = filename
            self.__init_data()


    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    #*#*#*#*#*#*# PUBLIC METHODS *#*#*#*#*#*#*#
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

    def get_snippets(self,row_indices,snippet_length,savefile=''):
        """
        Returns data snippets at specified row indices of length snippet_length
        Returned data is numpy array with dimension (len(row_indices),snippet_length,N),
        where N is the dimensionality of the data in provided csv file

        :par row_indices:       list of valid indices where data should be extracted (indices start from 0)
        :par snippet_length:    length of extracted data snippets
        :par savefile:          string with filename to destination if data is to be saved (if specified, data will be saved). Extention will automatically be .p

        :returns:               list containing [numpy array with data , savefile path (if given , else excluded)]

        """

        self.__output_data = np.zeros((len(row_indices),snippet_length,self.__row_length))

        assert all(value < self.__row_count for value in row_indices)

        i = 0
        for row_id in row_indices:
            row = self.__pd_values[row_id:row_id + snippet_length]

            self.__output_data[i] = row
            i += 1

        if savefile:
            savefile = os.path.splitext(savefile)[0] + '.p'
            pickle.dump(self.__output_data,open(savefile,'wb'))

            print(f"Saved data to {savefile}")
            
            return self.__output_data , savefile

        return self.__output_data

    
    def get_snippets_pre_post(self,row_indices,n_before,n_after):
        """    
        Returns data snippets at specified row indices, n_before rows before and n_after rows after each specified row 
        Returned data is numpy array with dimension (len(row_indices),n_before + n_after + 1, N),
        where N is the dimensionality of the data in provided csv file.

        min indice - n_before and max indice + n_after must be within data size

        :par row_indices:       list of valid indices where data should be extracted (indices start from 0)
        :par n_before:          number of rows before each indice to include
        :par n_after:           number of rows after each indice to include

        :returns:               numpy array with data
        """
        
        self.__output_data = np.zeros((len(row_indices),n_before + n_after + 1,self.__row_length))

        assert all( (value + n_after) < self.__row_count for value in row_indices), f"Some given index + {n_after} exceeds row count ({self.__row_count})"
        assert all(0 <= (value - n_before) for value in row_indices), f"Some given index - {n_before} negative"

        i = 0
        for row_id in row_indices:
            row = self.__pd_values[row_id - n_before : row_id + n_after + 1]

            self.__output_data[i] = row
            i += 1

        return self.__output_data


    def get_random_snippets(self,n_snippets,snippet_length,excluded_values=None,allow_overlap=False,savefile=''):
        """
        Creates n_snippets of length snippet_length randomly chosen parts of the data.

        :par n_snippets:        number of data snippets
        :par snippet_length:    length of snippets (number of frames)
        :par excluded_values:   list of values to be excluded from sampling. Disallows interval starting at excluded value and extending over snippet_length rows
        :par allow_overlap:     allow extracted data snippets to overlap (excluded_values will not be overlapped)
        :par savefile:          string with filename to destination if data is to be saved (if specified, data will be saved). Extention will automatically be .p

        :returns:               list containing [data in numpy array , list of starting indices of each returned rows] , savepath (if given , else excluded)]
        """

        if excluded_values is None:
            excluded_values = []

        self.__n_snippets = n_snippets
        self.__snippet_length = snippet_length
        self.__illegal_values = excluded_values

        self.__init_pars()

        if self.verbose:
            print(f"Allow overlap: {allow_overlap}")
            print(f"Number of snippets: {n_snippets}")
            print(f"Snippet length: {snippet_length}")
            print(f"Excluded values: {excluded_values}")
            print(f"# Available values: {len(self.__available_rows)}")

        if not self.__assert_data():
            return [] , 0 , 0
        
        while len(self.__chosen_rows) < self.__n_snippets:
            
            if len(self.__available_rows) > 0:
                choice = np.random.choice(self.__available_rows)
            else:
                # If poor random choices are made it can result in a choice blocking us from choosing n_snippets snippets
                # even though this is possible through some combination.
                # If so, we reinitialize and try again and hope for a better seed. 
                # This can only occur when size of data is close to n_snippets*snippet_length and overlap is false.

                self.__init_pars()
                continue
                
            
            if choice in self.__available_rows:
                try:
                    if allow_overlap: 
                        self.__illegal_rows.update([choice])
                    else:
                        # All rows which would result in overlapping data are added as illegal intervals
                        illegal_interval = list(range( max(0 , choice-self.__snippet_length+1) , min(choice + self.__snippet_length , self.__row_count - 1) ))
                        self.__illegal_rows.update(illegal_interval)

                    
                    self.__chosen_rows.append(choice)

                    row = self.__pd_values[choice:choice+self.__snippet_length]

                    self.__output_data[self.__i] = row

                    self.__available_rows = np.array([v for v in self.__available_rows if v not in self.__illegal_rows])

                    self.__i += 1

                except IndexError:
                    print(f"Indexerror {self.__snippet_length}")
                    print(f"Max val {len(self.__pd_values)}")
                    pass

        
        if savefile:
            savefile = os.path.splitext(savefile)[0] + '.p'
            pickle.dump(self.__output_data,open(savefile,'wb'))

            print(f"Saved data to {savefile}")

            return self.__output_data , self.__chosen_rows , savefile

        return self.__output_data , self.__chosen_rows


    def get_dataframe(self):
        """
        Return entire csv file as pandas dataframe
        """
        return self.__pd_data


    def get_data(self):
        """
        Return entire csv file as numpy array
        """
        return self.__pd_values


    

    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    #*#*#*#*#*#*# PRIVATE METHODS *#*#*#*#*#*#*#
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#


    def __init_pars(self):
        """
        Initializes various parameters used for valid random sampling
        """
        self.__chosen_rows = []
        self.__illegal_rows = set()
        
        # Adds manually fed excluded values
        [self.__illegal_rows.update(list(range(     max(0 , ill_val-self.__snippet_length + 1),
                                                    min(ill_val + self.__snippet_length , self.__row_count - 1)))) for ill_val in self.__illegal_values]

        # Pre-allocate matrix
        self.__output_data = np.zeros((self.__n_snippets,self.__snippet_length,self.__row_length))
        self.__available_rows = np.array([v for v in np.arange(self.__row_count - self.__snippet_length + 1) if v not in self.__illegal_rows])

        self.__i = 0


    def __assert_data(self):
        """
        Assert that there is enough data to produce the wanted number of snippets
        """

        if ( self.__n_snippets > len(self.__available_rows) ) or (self.__n_snippets*self.__snippet_length > self.__row_count - len(self.__illegal_rows)):
            print(f"Not enough data ({self.__row_count}) to return {self.__n_snippets} snippets of length {self.__snippet_length}")

            if self.verbose:
                print(f"Available values ({len(self.__available_rows)}): {self.__available_rows}")
                print(f"Excluded values: {self.__illegal_values} +- {self.__snippet_length - 1}")

            return False
        else:
            return True

    
    def __init_data(self):
        """
        Reads given CSV file into a pandas dataframe and produces some stats used in extraction
        """

        self.__pd_data = pd.read_csv(self.__filename,sep=',',header=self.__n_header_lines - 1) # n_headers = 0 will automatically read header if one exists
        self.__pd_values = self.__pd_data.values
        self.__row_count , self.__row_length = np.shape(self.__pd_values)

        if self.verbose:
            print(f"Row length assumed to be uniformly: {self.__row_length}")
            print(f"Data contains {self.__row_count} rows of data")


    

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
#*#*#*#*#*#*# OTHER FUNCTIONS #*#*#*#*#*#*#
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(filename, n_snippets,snippet_length,savefile,allow_overlap,n_header_lines,excluded_values,verbose):
    snipper = DataSnipper(filename,n_header_lines,verbose)
    data , rows , n_snips = snipper.get_random_snippets(n_snippets, snippet_length, excluded_values, allow_overlap,savefile)

    print(data)

    

if __name__ == "__main__":

    CLI=argparse.ArgumentParser()

    CLI.add_argument(
        "--filename",
        type=str,
        help='Full path to csv file'
    )
    CLI.add_argument(
        "--n_snippets",
        type=int,
        help='Number of data partitions to return'
    )
    CLI.add_argument(
        "--snippet_length",
        type=int,
        help='Length of returned data partition'
    )
    CLI.add_argument(
        "--n_header_lines",
        type=int,
        default=1,
        help='Number of rows of headers in csv file'
    )
    CLI.add_argument(
        "--allow_overlap",
        type=str2bool,
        default=False,
        help='Allows returned partitions to overlap'
    )
    CLI.add_argument(
        "--savefile",
        type=str,
        default='',
        help='Name of savefile if resulting data is to be saved (to pickle)'
    )
    CLI.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help='Print more explicit output'
    )
    CLI.add_argument(
        "--excluded_values",
        nargs='+',
        type=int,
        help='Option to add values which are not to be chosen as partition'
    )

    args = CLI.parse_args()
    filename = args.filename
    n_snippets = args.n_snippets
    snippet_length = args.snippet_length
    n_header_lines = args.n_header_lines
    allow_overlap = args.allow_overlap
    savefile = args.savefile
    verbose = args.verbose
    excluded_values = args.excluded_values

    main(filename, n_snippets,snippet_length,savefile,allow_overlap,n_header_lines,excluded_values,verbose)
