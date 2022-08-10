# OUTPUT

# results_file = "Output_Data\Results_Raw.txt"
results_file = "Output_Data\Results_CldCmp.txt"
# results_file = "Output_Data\Results_PNet.txt"

img_path = "Output\\"
def write_results_to_file(results):
    print("...writing")
    file = open(results_file, 'a')
    file.write(results + "\n")
    file.close()