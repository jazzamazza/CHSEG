python.exe RunExperiment.py cure raw 0.125 162 750
rm .\Data\Clustered\cure\*.npy
#python.exe RunExperiment.py cure geo 0.200 106 750
#rm .\Data\Clustered\cure\*.npy
python.exe RunExperiment.py cure pnet 0.100 100 750
rm .\Data\Clustered\cure\*.npy