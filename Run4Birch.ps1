echo 'Running Birch on geo - 50 clusters'
echo 'python RunExperiment.py birch raw 0.075 50 50'
python RunExperiment.py birch pnet 0.095 50 50

echo 'Running Birch on geo - 300 clusters'
echo 'python RunExperiment.py birch raw 0.075 300 300'
python RunExperiment.py birch pnet 0.095 300 300

echo 'Running Birch on geo - 425 clusters'
echo 'python RunExperiment.py birch raw 0.075 425 425'
python RunExperiment.py birch pnet 0.095 425 425

echo 'Running Birch on geo - 550 clusters'
echo 'python RunExperiment.py birch raw 0.075 550 550'
python RunExperiment.py birch pnet 0.095 550 550
