from Experiment import Experiment
from Experiment import Graph
from os.path import exists
from sys import argv

class ExperimentConfig():
    
    def config_experiment(self, algs = 'all', input_files = 'all', clust_start = 50, clust_end = 500, ds_amt = 0.0):
        #set defaults here
        DATA_ROOT = './Data/'
        FILE_NAME_BASE = 'church_registered'
        DEFAULT_EXT = '.npy'
        DS_APPEND = '_ds_' + str(ds_amt)
        RAW_DEFAULT = DATA_ROOT + FILE_NAME_BASE + DS_APPEND + DEFAULT_EXT
        GEO_DEFAULT = DATA_ROOT + 'CC/' + FILE_NAME_BASE + DS_APPEND + '_cc_23_feats.las'
        PNET_DEFAULT = DATA_ROOT + 'PNet/' + FILE_NAME_BASE + DS_APPEND +'_pnet_all_fix' + DEFAULT_EXT
        
        files = {'all': [RAW_DEFAULT, GEO_DEFAULT, PNET_DEFAULT],
                 'raw': [RAW_DEFAULT],
                 'geo': [GEO_DEFAULT],
                 'pnet': [PNET_DEFAULT]}
        
        algos = {'all': ["kmeans", "birch", "cure", "aggl"],
                 'kmeans': ['kmeans'],
                 'birch': ['birch'],
                 'cure': ['cure'],
                 'aggl': ['aggl']}
        
        self.algs = algos[algs]
        self.in_files = files[input_files]
        ds_str = str(ds_amt)
        self.output_file = "./Results/test_ds_" + ds_str + '_algs_' + algs + '_files_' + input_files + '.csv' 
        self.clust_start = clust_start
        self.clust_end = clust_end
    
    def config_graph(self, csv_file, metric, alg, title):
        self.csv_file = csv_file
        self.metric = metric
        self.alg = alg
        self.title = title
        

if __name__ == "__main__":
    assert(len(argv) == 6), f"Five (5) arguments expected, got: {len(argv)-1}"
    algs = argv[1]
    in_files = argv[2]
    ds_amt = argv[3]
    clust_start = int(argv[4])
    clust_end = int(argv[5])
    
        
    my_config = ExperimentConfig()
    my_config.config_experiment(algs, in_files, clust_start, clust_end, ds_amt)
    my_experiment = Experiment()
    my_experiment.run_experiment(my_config.clust_start, my_config.clust_end, my_config.algs, my_config.in_files, my_config.output_file)
