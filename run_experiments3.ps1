python.exe RunExperiment.py cure geo 0.075 100 750
rm .\Data\Clustered\cure\*.npy
python.exe RunExperiment.py cure raw 0.075 100 750
rm .\Data\Clustered\cure\*.npy
python.exe RunExperiment.py cure pnet 0.075 100 750
rm .\Data\Clustered\cure\*.npy