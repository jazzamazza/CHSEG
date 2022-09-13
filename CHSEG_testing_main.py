from Clustering import Clustering
from PointCloudLoader import PointCloudLoader
from Classification import Classification
from Metrics import Testing
from Outputting import write_results_to_file

class CHSEG_TESTING_main:
    def __init__(self):
        '''Initialise class variables'''
        # loading point cloud variables
        self.pointCloud, self.pcd_with_truths = None, None
        self.ds = False
        self.dsSize = 0

        # classifying point cloud variables
        self.classifier = Classification()
        self.testing = None
    
    def class_and_eval(self, unique_labels, y_km):
        '''Method responsible for classifying point cloud and evaluating the classification results
        args:
            unique_labels: the unique labels of all the clusters
            y_km: the produced clusters'''
        # Classification
        self.classifier.classify(unique_labels, y_km, self.pcd_with_truths, self.index)
        true_labels, predicted_labels = self.classifier.get_ground_truth()

        # Evaluation
        userInput = input("\nEvaluate Results (y/n)?")
        if (userInput == "q"): return 0
        elif (userInput=="y"): self.testing.classification_metrics(true_labels, predicted_labels)
      
    def setup(self, option):
        '''Helper method to create PointCloudLoader object and call method to load downsampled point cloud files
        args:
            option: point cloud dataset to load from file
        '''
        loader = PointCloudLoader()
        self.set_truth_label_idx(option)
        if (option == "1"): 
            name, path1, path2, div255 = "*************Raw Point Cloud*************", "Data\\raw\FINAL-PCD_raw_0.085.npy", "Data\\raw\FINAL-PCD-ALL_raw_0.085.npy", False
        elif (option == "2"): 
            name, path1, path2, div255 = "*************Point Cloud with Cloud Compare Features*************", "Data\cldCmp\FINAL-PCD_cloudCompare_0.085.npy", "Data\cldCmp\FINAL-PCD-ALL_PCD_cloudCompare_0.085.npy", True
        elif (option == "3"):
            name, path1, path2, div255 = "*************Point Cloud with PointNet++ Features*************", "Data\pnet\FINAL-PCD_WORKING_pointnet_0.19.npy", "Data\pnet\FINAL-PCD-ALL_PCD_WORKING_pointnet_0.19.npy", False

        write_results_to_file(name)
        self.pointCloud, self.pcd_with_truths = loader.load_ds_from_file(path1, path2, self.index, div255)
        self.testing = Testing(self.pointCloud)

    def set_truth_label_idx(self, pcd_choice):
        '''Stores the truth label index of the point cloud in a class variable
        args:
            pcd_choice: point cloud dataset to load
        '''
        if pcd_choice == "1": self.index = 4 # raw point cloud
        else: self.index = 3 # cloud compare and PointNet++ point cloud
        write_results_to_file("Ground Truth Index:" + str(self.index))

    def application(self):
        '''Interative Application, responsible for getting user input'''
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
                                "\n r : Restart the Application\n")
                if (userInput == "q"): break
                
                elif (userInput == "0"): u_lbl, lbl = clustering.k_means_clustering(13)
                elif (userInput == "1"): u_lbl, lbl = clustering.dbscan_clustering()
                elif (userInput == "2"): u_lbl, lbl = clustering.optics_clustering()
                elif (userInput == "3"): u_lbl, lbl = clustering.mean_shift_clustering()

                # classify point cloud and evaluate classification
                userInput = input("\nClassify Clustering Result (y/n)?")
                if (userInput == "q"): break
                elif (userInput=="y"): 
                    x = self.class_and_eval(u_lbl, lbl)
                    if x==0: break
            
if __name__=="__main__":
    main = CHSEG_TESTING_main()
    main.application()