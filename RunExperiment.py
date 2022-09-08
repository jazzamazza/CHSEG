from Experiment import Experiment
from Experiment import Graph

class ExperimentConfig():
    def __init__(self) -> None:
        self.algs = ["kmeans", "birch", "aggl", "cure"]
        self.output_file = "./Results/test_pnet_all_new.csv"

if __name__ == "__main__":
    my_config = ExperimentConfig()
    my_experiment = Experiment()
    # my_graph = Graph("./Results/test_100_439.csv")
    my_experiment.run_experiment(10, 50, [my_config.algs[2]], ["./Data/PNet/church_registered_ds_0.075x0.085x0.1_pnet.npy"])