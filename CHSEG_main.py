from Clustering import Clustering
from PointCloudLoader import PointCloudLoader
from Classification import Classification
from Metrics import Testing
from Outputting import write_results_to_file

class CHSEG_main:
    def __init__(self, pcd_path, ds_pcd_file_path, ds_pcd_all_file_path, pnet_input_file_path):
        '''Initialise class variables'''
        # loading point cloud variables
        self.pcd_file_path = pcd_path
        self.ds_pcd_file_path = ds_pcd_file_path
        self.ds_pcd_all_file_path = ds_pcd_all_file_path
        self.pnet_input_file_path = pnet_input_file_path
        self.pointCloud, self.pcd_with_truths = None, None
        self.ds = False
        self.dsSize = 0
        self.load_downsampled = False

        self.names = {1:'raw', 2:'cldCmp', 3:'pnet'}

        print("_____________---tsting_________________")
        print("keys:", self.names[1])

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
        self.set_truth_label_idx(option)
        pc_loader = PointCloudLoader(self.pcd_file_path, self.ds, self.dsSize, self.index)
        if (option == "4"): 
            self.pointCloud = pc_loader.load_pointnet_pcd(pnet_input_file_path) # setup point cloud with PointNet++ features
        else:
            self.pointCloud, self.pcd_with_truths = pc_loader.load_point_cloud(self.names[int(option)], self.load_downsampled, self.ds_pcd_file_path, self.ds_pcd_all_file_path)
        # if (option == "1"): 
        #     write_results_to_file("*************Raw Point Cloud*************")
        #     self.pointCloud, self.pcd_with_truths = pc_loader.load_point_cloud(self.names[int(option)], self.load_downsampled, self.ds_pcd_file_path, self.ds_pcd_all_file_path) # setup point cloud with raw features 
        # elif (option == "2"): 
        #     write_results_to_file("*************Point Cloud with Cloud Compare Features*************")
        #     self.pointCloud, self.pcd_with_truths = pc_loader.load_point_cloud('cldCmp') # setup point cloud with Cloud Compare features
        # elif (option == "3"):
        #     write_results_to_file("*************Point Cloud with PointNet++ Features*************")
        #     self.pointCloud, self.pcd_with_truths = pc_loader.load_point_cloud('pnet')
        # elif (option == "4"): 
        #     self.pointCloud = pc_loader.load_pointnet_pcd() # setup point cloud with PointNet++ features
        self.testing = Testing(self.pointCloud)
        write_results_to_file("Downsample Size:" + str(self.dsSize))

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
            
            if (pcd_choice!="4"):
                userInput = input("\nLoad an already Downsampled Point Cloud (y/n)?")
                if (userInput == "q"): break
                elif (userInput=="y"): 
                    self.load_downsampled = True
                else:
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
    pcd_file_path = "Data\church_registered.npy" # choose raw, cloud compare or pointnet un-downsampled file

    ds_pcd_file_path = "Data\\raw\FINAL-PCD_raw_0.085.npy" # choose raw, cloud compare or pointnet downsampled finalpcd file
    ds_pcd_all_file_path = "Data\\raw\FINAL-PCD-ALL_raw_0.085.npy" # choose raw, cloud compare or pointnet downsampled finalpcdAll file

    pnet_input_file_path = "Data\church_registered_downsampled_0.075.ply" # pre-downsampled and restuctured raw point cloud file for inputting into PointNett

    main = CHSEG_main(pcd_file_path, ds_pcd_file_path, ds_pcd_all_file_path, pnet_input_file_path)
    main.application()
