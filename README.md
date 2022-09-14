# UCT Honours Project - (CHSEG) Unsupervised Semantic Segmentation of Cultural Heritage Sites

## Authors: Jared May, (Leah Gluckman, Jemma Sundelson)

_

## How to set up


### Create folders and extract data
Create folders in Root Dir:
 - Data
 - Plots
 - Results

In Data create folders:
- CC
- Pnet
- Clustered

In Clustered create folders:
- aggl
- birch
- kmeans
- cure

Extract PNet and CC zips into their respective folders.
Extract data .zip into Data root.

### Install requirements

Python 3.6 required. NB.
Using [venv and pip] or [conda and pip] install requirements.txt (CUDA capable GPU required) or requirements_mac.txt (for non CUDA GPU)
Note:
One cannot run PointNet++/pytorch on a dataset without a CUDA GPU.
On linux ./Scripts/fixpptk.sh may have to be used to fix PPTK.

## How to Run

### Experiments

Use the included scripts in Scripts as a demo on Windows/MacOS/linux.

or

python RunExperiment.py [alg] [dataset] [downsample amount] [clusters start] [clusters end]

Otherwise run CHSEG_main.py which is an interactive self explanatory GUI. (Note this script has not been maintained since an early version of the project in limited testing it still works fine.)

Experiments can also be setup to run manually in Experiments.py (requires editing bottom of file to be able to run as main).

### Tools and other fiddling

Tools.py offers many tools to create Pointnet++ ready datasets, fix broken result files if they occur, run Pointnet++ model on a dataset and other well tools. It does however require going into the script and manually editing file paths. But it is menu driven and largely self explanatory.
