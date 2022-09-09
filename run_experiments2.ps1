python.exe RunExperiment.py kmeans raw 0.050 100 750
rm .\Data\Clustered\kmeans\*.npy
python.exe RunExperiment.py kmeans geo 0.050 100 750
rm .\Data\Clustered\kmeans\*.npy
python.exe RunExperiment.py kmeans pnet 0.050 100 750
rm .\Data\Clustered\kmeans\*.npy

python.exe RunExperiment.py birch raw 0.075 100 750
rm .\Data\Clustered\birch\*.npy
python.exe RunExperiment.py birch geo 0.075 100 750
rm .\Data\Clustered\birch\*.npy
python.exe RunExperiment.py birch pnet 0.075 100 750
rm .\Data\Clustered\birch\*.npy