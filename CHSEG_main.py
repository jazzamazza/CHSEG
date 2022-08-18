from Clustering import Clustering
from PointCloudLoader import PointCloudLoader
from Classification import Classification
from Metrics import Testing
from Outputting import write_results_to_file

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

    def class_and_eval(self, unique_labels, y_km, file_name):
        # Classification
        t = self.pcd_with_truths
        self.classifier.classify(unique_labels, y_km, t, self.index, self.class_pcd_file_path, file_name)
        true_labels, predicted_labels = self.classifier.get_ground_truth()
        # visualise classification in PPTK
        # self.classifier.visualise_classification(self.pointCloud)

        # Evaluation
        userInput = input("\nEvaluate Results (y/n)?")
        if (userInput == "q"): return 0
        elif (userInput=="y"): 
            self.testing.classification_metrics(true_labels, predicted_labels)

    # Helper method to call method to load point cloud files  
    # Returns a PointCloud in a numpy array      
    def setup(self, option):
        pc_loader = PointCloudLoader(self.pcd_file_path)
        if (option == "1"): 
            write_results_to_file("*************Raw Point Cloud*************")
            self.pointCloud, self.pcd_with_truths = pc_loader.load_point_cloud_npy(self.vis, self.ds, self.dsSize) # setup point cloud with raw features 
        elif (option == "2"): 
            write_results_to_file("*************Point Cloud with Cloud Compare Features*************")
            self.pointCloud, self.pcd_with_truths = pc_loader.load_point_cloud_las(self.vis, self.ds, self.dsSize) # setup point cloud with Cloud Compare features
        elif (option == "3"):
            write_results_to_file("*************Point Cloud with PointNet++ Features*************")
            self.pointCloud, self.pcd_with_truths = pc_loader.load_point_cloud_pNet_npy(self.vis, self.ds, self.dsSize)
        elif (option == "4"): 
            self.pointCloud = pc_loader.loadPointCloud_pNet(self.vis) # setup point cloud with PointNet++ features
        self.set_truth_label_idx(option)
        self.testing = Testing(self.pointCloud)
        write_results_to_file("Downsample Size:" + str(self.dsSize))

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
            
            # Setup and visualise point cloud based on user input
            userInput = input("\nVisualise Point Cloud (y/n)?")
            if (userInput == "q"): break
            elif (userInput=="y"): self.vis = True
            
            if (pcd_choice!="4"):
                userInput = input("\nDownsample Point Cloud (y/n)?")
                if (userInput == "q"): break
                elif (userInput=="y"): 
                    self.ds = True
                    userInput = input("\nSpecify Downsample Size (0.5, 1, 2, etc.)?")
                    if (userInput == "q"): break
                    self.dsSize = float(userInput)

            self.setup(pcd_choice)
            clustering = Clustering(self.pointCloud , pcd_choice)
            
            while (userInput != "r" and userInput != "q"):
                write_results_to_file("--------------------------------------------------------")
                # cluster point cloud    
                userInput = input("\nChoose Clustering Method(s):"+
                                "\n 0 : K-Means Clustering" +
                                "\n 1 : DBSCAN Clustering"+
                                "\n 2 : OPTICS Clustering"+
                                "\n 3 : Mean-Shift Clustering"+
                                "\n r : Restart the Application\n")
                if (userInput == "q"): break
                
                elif (userInput == "0"): u_lbl, lbl, f_name = clustering.k_means_clustering(13)
                elif (userInput == "1"): u_lbl, lbl, f_name = clustering.dbscan_clustering()
                elif (userInput == "2"): u_lbl, lbl, f_name = clustering.optics_clustering()
                elif (userInput == "3"): u_lbl, lbl, f_name = clustering.mean_shift_clustering()

                # classify point cloud and evaluate classification
                userInput = input("\nClassify Clustering Result (y/n)?")
                if (userInput == "q"): break
                elif (userInput=="y"): 
                    x = self.class_and_eval(u_lbl, lbl, f_name)
                    if x==0: break
            
if __name__=="__main__":
    # Raw data
    pcd_file_path = "Data\church_registered.npy"
    classified_pcd_path = "Output_Data\\raw" 

    # Cloud Compare data
    # pcd_file_path = "Data\church_registered_cc_raw.las"
    # classified_pcd_path = "Output_Data\cldCmp"

    # PointNet++ data
    # pcd_file_path = "Data\church_registered_pnet_wtruth_0.05.ply"
    # classified_pcd_path = "Output_Data\pnet"

    main = CHSEG_main(pcd_file_path, classified_pcd_path)
    main.application()
