from Clustering import Clustering
from PointCloudLoader import PointCloudLoader
from Classification import Classification
from Metrics import Testing
from Outputting import write_results_to_file
import numpy as np

class CHSEG_main:
    def __init__(self, pcd_path, classified_pcd_path):
          # loading point cloud variables
          self.pcd_file_path = pcd_path
          self.pointCloud = None
          self.pcd_with_truths = None
          self.vis = False
          self.ds = False
          self.dsSize = 0

          # classifying point cloud variables
          self.classifier = Classification()
          self.testing = None
          self.class_pcd_file_path = classified_pcd_path

    ################################################################################################
    ######LOADING ALREADY DOWNSAMPLED PCDS###########
    def load_raw_from_file(self, path1, path2):
        print("\n** Loading Point Cloud FINAL-PCD_raw_0.085**")
        loader = PointCloudLoader()
        final_pcd = np.load(path1)
        loader.get_attributes(final_pcd)   
        
        # divide point_cloud into points and features 
        points = final_pcd[:,:3]
        intensity = final_pcd[:,3:4]
        print("final_pcd points:", points)
        print("final_pcd intensity:", intensity)
        print("final_pcd[0]:", final_pcd[0])

        print("\n** Loading Point Cloud FINAL-PCD-ALL_raw_0.085**")
        final_pcd_all = np.load(path2)
        loader.get_attributes(final_pcd_all)   
        
        # divide point_cloud into points and features 
        points = final_pcd_all[:,:3]
        intensity = final_pcd_all[:,3:4]
        truth_label = final_pcd_all[:,4:5]
        print("initial truth_label:", truth_label)
        final_pcd_all[:,4:5] = np.ceil(final_pcd_all[:,4:5])
        print("final_pcd_all points:", points)
        print("final_pcd_all intensity:", intensity)
        print("final_pcd_all truth_label:", final_pcd_all[:,3:4])
        print("final_pcd_all[0]:", final_pcd_all[0])

        return final_pcd, final_pcd_all
    
    def load_pointnet_from_file(self, path1, path2):
        print("\n** Loading Point Cloud FINAL-PCD_pointnet_0.085**")
        loader = PointCloudLoader()
        final_pcd = np.load(path1)
        loader.get_attributes(final_pcd)   
        
        # divide point_cloud into points and features 
        points = final_pcd[:,:3]
        features = final_pcd[:,3:]
        print("final_pcd points:", points)
        print("final_pcd features:", features)
        print("final_pcd[0]:", final_pcd[0])

        print("\n** Loading Point Cloud FINAL-PCD-ALL_pointnet_0.085**")
        final_pcd_all = np.load(path2)
        loader.get_attributes(final_pcd_all)   
        
        # divide point_cloud into points and features 
        points = final_pcd_all[:,:3]
        final_pcd_all[:,3:4] = np.ceil(final_pcd_all[:,3:4])
        features = final_pcd_all[:,4:]
        print("final_pcd_all points:", points)
        print("final_pcd_all truth_label:", final_pcd_all[:,3:4])
        print("final_pcd_all features:", features)
        print("final_pcd_all[0]:", final_pcd_all[0])
        
        return final_pcd, final_pcd_all
    
    def load_cloudCompare_from_file(self, path1, path2):
        print("\n** Loading Point Cloud FINAL-PCD_cloudcompare_0.085**")
        loader = PointCloudLoader()
        final_pcd = np.load(path1)
        loader.get_attributes(final_pcd)   
        
        # divide point_cloud into points and features 
        points = final_pcd[:,:3]
        features = final_pcd[:,3:]
        print("final_pcd points:", points)
        print("final_pcd features:", features)
        print("final_pcd[0]:", final_pcd[0])

        print("\n** Loading Point Cloud FINAL-PCD-ALL_cloudcompare_0.085**")
        final_pcd_all = np.load(path2)
        loader.get_attributes(final_pcd_all)   
        
        # divide point_cloud into points and features 
        points = final_pcd_all[:,:3]
        final_pcd_all[:,3:4] = np.ceil(final_pcd_all[:,3:4]/255)
        features = final_pcd_all[:,4:]
        print("final_pcd_all points:", points)
        print("final_pcd_all truth_label:", final_pcd_all[:,3:4])
        print("final_pcd_all features:", features)
        print("final_pcd_all[0]:", final_pcd_all[0])

        # for i in truth_label:
        #     if i[0] != float(0):
        #         print(i)
        
        return final_pcd, final_pcd_all

    def classify_test(self, u_lbl, lbl, t, f_name):
        # classify point cloud and evaluate classification
        userInput = input("\nClassify Clustering Result (y/n)?")
        if (userInput == "q"): return 0
        elif (userInput=="y"): 
            x = self.class_and_eval(u_lbl, lbl, t, f_name)
            if x==0: return 0
        return 1
    ################################################################################################

    def class_and_eval(self, unique_labels, y_km, file_name):
        # Classification
        t = self.pcd_with_truths
        self.classifier.classify(unique_labels, y_km, t, self.index, self.class_pcd_file_path, file_name)
        true_labels, predicted_labels = self.classifier.get_ground_truth()

        # Evaluation
        userInput = input("\nEvaluate Results (y/n)?")
        if (userInput == "q"): return 0
        elif (userInput=="y"): 
            print("here true labels:", true_labels)
            print("here predicted_labels:", predicted_labels)
            self.testing.classification_metrics(true_labels, predicted_labels)

    # Helper method to call method to load point cloud files  
    # Returns a PointCloud in a numpy array      
    def setup(self, option):
        if (option == "1"): 
            write_results_to_file("*************Raw Point Cloud*************")
            self.pointCloud, self.pcd_with_truths = self.load_raw_from_file("Data\\raw\FINAL-PCD_raw_0.085.npy", "Data\\raw\FINAL-PCD-ALL_raw_0.085.npy") # setup point cloud with raw features 
        elif (option == "2"): 
            write_results_to_file("*************Point Cloud with Cloud Compare Features*************")
            self.pointCloud, self.pcd_with_truths = self.load_cloudCompare_from_file("Data\cldCmp\FINAL-PCD_cloudCompare_0.085.npy", "Data\cldCmp\FINAL-PCD-ALL_PCD_cloudCompare_0.085.npy")
        elif (option == "3"):
            write_results_to_file("*************Point Cloud with PointNet++ Features*************")
            self.pointCloud, self.pcd_with_truths = self.load_pointnet_from_file("Data\pnet\FINAL-PCD_pointnet_0.085_REDO.npy", "Data\pnet\FINAL-PCD-ALL_PCD_pointnet_0.085_REDO.npy")

        self.set_truth_label_idx(option)
        self.testing = Testing(self.pointCloud)

    def set_truth_label_idx(self, pcd_choice):
        if pcd_choice == "1": self.index = 4 # raw point cloud
        if pcd_choice == "2": self.index = 3 # cloud compare point cloud
        if pcd_choice == "3": self.index = 3 # PointNet++ point cloud
        write_results_to_file("Ground Truth Index:" + str(self.index))

    # interactive application
    def application(self):
        userInput = ""
        while (userInput != "q"):
            print("--------------Welcome---------------")
            print("Type q to quit the application")
            # Choose Point Cloud
            userInput = input("\nChoose Point Cloud Input:"+
                            "\n 1 : Point Cloud with Raw Features"+
                            "\n 2 : Point Cloud with Cloud Compare Features"+
                            "\n 3 : Point Cloud with PointNet++ Features"+
                            "\n 4 : Create Point Cloud with PointNet++ Features\n")
            if (userInput == "q"): break
            pcd_choice = userInput

            self.setup(pcd_choice)
            clustering = Clustering(self.pointCloud, pcd_choice)
            
            while (userInput != "r" and userInput != "q"):
                write_results_to_file("--------------------------------------------------------")
                # cluster point cloud    
                userInput = input("\nChoose Clustering Method(s):"+
                                "\n 0 : K-Means Clustering" +
                                "\n 1 : DBSCAN Clustering"+
                                "\n 2 : OPTICS Clustering"+
                                "\n 3 : Mean-Shift Clustering"+
                                "\n 4 : All Clusterings"+
                                "\n r : Restart the Application\n")
                if (userInput == "q"): break
                
                elif (userInput == "0"): u_lbl, lbl, f_name = clustering.k_means_clustering(13)
                elif (userInput == "1"): u_lbl, lbl, f_name = clustering.dbscan_clustering()
                elif (userInput == "2"): u_lbl, lbl, f_name = clustering.optics_clustering()
                elif (userInput == "3"): u_lbl, lbl, f_name = clustering.mean_shift_clustering()
                elif (userInput == "4"): 
                    u_lbl, lbl, f_name = clustering.k_means_clustering(13)
                    x = self.classify_test(u_lbl, lbl, f_name)
                    if x==0: break
                    u_lbl, lbl, f_name = clustering.dbscan_clustering()
                    x = self.classify_test(u_lbl, lbl, f_name)
                    if x==0: break
                    u_lbl, lbl, f_name = clustering.optics_clustering()
                    x = self.classify_test(u_lbl, lbl, f_name)
                    if x==0: break
                    u_lbl, lbl, f_name = clustering.mean_shift_clustering()
                    x = self.classify_test(u_lbl, lbl, f_name)
                    break

                # classify point cloud and evaluate classification
                userInput = input("\nClassify Clustering Result (y/n)?")
                if (userInput == "q"): break
                elif (userInput=="y"): 
                    x = self.class_and_eval(u_lbl, lbl, f_name)
                    print("yes")
                    if x==0: break
            
if __name__=="__main__":
    # Raw data
    pcd_file_path = "Data\church_registered.npy"
    # classified_pcd_path = "Output_Data\\raw_tues_9_56" 

    # Cloud Compare data
    # pcd_file_path = "Data\church_registered_cc_raw.las"
   
    # pcd_file_path = ''
    classified_pcd_path = "Output_Data\cldCmp_tues_18_00"

    # PointNet++ data
    # pcd_file_path = "Data\church_registered_pnet_wtruth_0.05.ply"
    # classified_pcd_path = "Output_Data\pnet_10"

    main = CHSEG_main(pcd_file_path, classified_pcd_path)
    main.application()