install: venv
	. venv/bin/activate; pip3 install -r requirements.txt
	pip3 install open3d

venv :
	test -d venv || python3 -m venv venv

win_install: win_venv
	venv\Scripts\activate
	pip3 install -r requirements.txt
	pip3 install open3d

win_venv :
	python -m venv venv

run:	venv
	python3 CHSEG_testing_main.py	
	
clean:
	rm -r venv
