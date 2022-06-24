install: venv
	. venv/bin/activate; pip3 install -r requirements.txt
	pip3 install open3d

venv :
	test -d venv || python3 -m venv venv

run:	venv
	python3 set_up.py	
	
clean:
	rm -r venv