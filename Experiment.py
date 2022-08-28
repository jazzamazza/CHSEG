from PointCloudLoader import PointCloudLoader

class Experiment:
    def __init__(self) -> None:
        self.dataset = None
        self.ds = None
        #self.alg
        self.pcloader = None
        self.pcd = None
        self.pcd_truth = None
    
    def load(self, file_path, dataset):
        self.pcloader = PointCloudLoader(file_path)
        self.dataset = dataset
        self.pcd, self.pcd_truth = self.pcloader.load_point_cloud()
    
    def cluster(self, alg):
        pass