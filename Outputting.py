'''Reponsible for Outputting Results'''

results_file = "Results_PointNet_Testing.txt"
img_path = "Output\\"

def write_results_to_file(results):
    '''Method to write results to file
    args:
        results: an array of strings to write to file'''
    print("...writing")
    file = open(results_file, 'a')
    file.write(results + "\n")
    file.close()