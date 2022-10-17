from Clustering import Clustering
from PointCloudLoader import PointCloudLoader
from classification import Classification
from Metrics import Testing

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

    def class_and_eval(self, unique_labels, y_km, t, file_name):
        # Classification
        self.classifier.classify(unique_labels, y_km, t, self.index, self.class_pcd_file_path, file_name)
        true_labels, predicted_labels = self.classifier.get_ground_truth()
        

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
            self.pointCloud, self.pcd_with_truths = pc_loader.load_point_cloud_npy(self.vis, self.ds, self.dsSize) # setup point cloud with raw features 
        elif (option == "2"): 
            self.pointCloud, self.pcd_with_truths = pc_loader.load_point_cloud_las_npy(self.vis, self.ds, self.dsSize) # setup point cloud with Cloud Compare features
        elif (option == "3"):
            self.pointCloud, self.pcd_with_truths = pc_loader.load_point_cloud_pNet_npy(self.vis, self.ds, self.dsSize)
        elif (option == "4"): 
            self.pointCloud, self.pcd_with_truths = pc_loader.loadPointCloud_pNet(self.vis) # setup point cloud with PointNet++ features
        
        self.set_truth_label_idx(option)
        self.testing = Testing(self.pointCloud)

    def set_truth_label_idx(self, pcd_choice):
        if pcd_choice == "1": self.index = 4 # raw point cloud
        if pcd_choice == "2": self.index = 3 # cloud compare point cloud
        if pcd_choice == "3": self.index = 3 # PointNet++ point cloud

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
            clustering = Clustering(self.pointCloud, self.pcd_with_truths , pcd_choice)
            test = Testing(self.pointCloud)
            
            while (userInput != "r" and userInput != "q"):
                # cluster point cloud with different chosen algortihms   
                userInput = input("\nChoose Clustering Method(s):"+
                                "\n 0 : K-Means Clustering" +
                                "\n 1 : GMM Clustering"+
                                "\n 2 : K-medians Clustering"+
                                "\n 3 : fuzzy c-means Clustering"+
                                "\n x : Skip to clustering metrics" +
                                "\n r : Restart the Application\n")
                if (userInput == "q"): break

                elif (userInput == "0"): u_lbl, lbl, t, f_name = clustering.k_means_clustering(1000)
                elif (userInput == "1"): u_lbl, lbl, t, f_name = clustering.GMM_clustering()
                elif (userInput == "2"): u_lbl, lbl, t, f_name = clustering.KMedians_clustering()
                elif (userInput == "3"): u_lbl, lbl, t, f_name = clustering.fuzzy_cmeans_clustering() 

                #evaluate cluster validity metrics for all algorithms
                userInput = input("\nget clustering metrics (y/n)?")
                if (userInput == "q"): break
                elif (userInput=="y"): 
                    userInput = input("\nChoose Testing Method(s):"+
                              "\n 0 : Silhouette Coefficent for Kmeans" +
                              "\n 1 : Silhouette Coefficent for GMM" +
                              "\n 2 : Silhouette Coefficent for Kmedians" +
                              "\n 3 : Silhouette Coefficent for CLARANS" +
                              "\n 4 : Silhouette Index for fuzzy c-means"+
                              "\n 5 : DB Index for kmeans"+
                              "\n 6 : DB Index for GMM"+
                              "\n 7 : DB Index for kmedians"+
                              "\n 8 : DB Index for fuzzy c-means"+
                              "\n 9 : BIC"+
                              "\n x : Skip to Classification metrics" +
                              "\n r : Restart the Application\n")
                if (userInput == "q"): break
                elif (userInput == "x"): test.classification_metrics()
                elif (userInput == "0"): test.silhouette_kmeans()
                elif (userInput == "1"): test.silhouette_GMM()
                elif (userInput == "2"): test.silhouette_kmedians()
                elif (userInput == "3"): test.silhouette_Clarans()
                elif (userInput == "4"): test.silhouette_fuzzy_cmeans()
                elif (userInput == "5"): test.db_index_kmeans()
                elif (userInput == "6"): test.db_index_GMM()
                elif (userInput == "7"): test.db_index_Kmedians()
                elif (userInput == "8"): test.db_index_fuzzy_cmeans()
                elif (userInput == "9"): test.BIC()

                # classify point cloud and evaluate classification
                userInput = input("\nClassify Clustering Result (y/n)?")
                if (userInput == "q"): break
                elif (userInput=="y"): 
                    x = self.class_and_eval(u_lbl, lbl, t, f_name)
                    if x==0: break
            
if __name__=="__main__":
    # raw data path 
    pcd_file_path = "data/church_registered.npy"
    classified_pcd_path = "data/church_registered"

    # cloud compare path: my chosen features (down sampled 0.085)
    # pcd_file_path = "/Users/A102178/Desktop/data/selected_cloud_compare_0.085.npy"
    # classified_pcd_path = "/Users/A102178/Desktop/data/selected_cloud_compare_0.085"

    # point net path downsampled 0.085 0.075
    # pcd_file_path = "/Users/A102178/Desktop/data/working_all_pnet_0.075x0.085.npy"
    # classified_pcd_path = "/Users/A102178/Desktop/data/working_all_pnet_0.075x0.085"

    main = CHSEG_main(pcd_file_path, classified_pcd_path)
    main.application()

