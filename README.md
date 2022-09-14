# Semantically Meaningful Density-Based Clustering of Cultural Heritage Sites

`make run` to run the program \
`make clean` to clean the environment

**LINUX:** \
`make install` to activate a virtual environment and install all dependencies \
`make venv` to create a virtual environment

**WINDOWS:** \
`make win_install` to activate a virtual environment and install all dependencies \
`make win_venv` to create a virtual environment

**FILE PATHS:** \
In CHSEG_main.py change `pcd_file_path` to the file path of non-downsampled point cloud stored in a npy file,
                 change `ds_pcd_file_path` to the file path of a downsampled point cloud containing no ground truth labels,
                 change `ds_pcd_all_file_path` to the file path of a downsampled point cloud containing ground truth labels,
