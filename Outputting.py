class Outputting:
    '''Reponsible for Outputting Results'''
    def __init__(self, results_file):
        self.results_file = results_file

    def write_results_to_file(self, results):
        '''Method to write results to file
        args:
            results: an array of strings to write to file'''
        print("...writing")
        file = open(self.results_file, 'a')
        file.write(results + "\n")
        file.close()
    
    def write_results(self, arrResults):
          '''Write results to a file
          args: 
               arrResults: the array of strings to write to file'''
          for r in arrResults:
               self.write_results_to_file(r)