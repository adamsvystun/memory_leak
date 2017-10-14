all:

setup:
	pip3 install -r requirements.txt

test: $(shell find test -type f)
	python3 -m unittest discover -s test -p '*.py'
