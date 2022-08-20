# OUTPUT

# results_file = "MY_THESIS_RESULTS\mtr_Raw\Results_Raw_Final_Thurs_14_10.txt"
results_file = "Results_PointNet_Sat.txt"
# results_file = "Results_Raw_Final_Thurs_14_10_testing_db.txt"

# results_file = "Output_Data\Results_PNet_testing_k.txt"
# results_file = "Output_Data\Results_CldCmp_wed_8_35.txt"
# results_file = "Output_Data\Results_PNet.txt"

img_path = "Output\\"
def write_results_to_file(results):
    print("...writing")
    file = open(results_file, 'a')
    file.write(results + "\n")
    file.close()