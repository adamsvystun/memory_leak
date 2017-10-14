all:

web: $(shell find web -type f)
	python3 -m web

gen: $(shell find gen -type f)
	python3 -m gen

setup:
	pip3 install -r requirements.txt

# test: $(shell find test -type f)
	# python3 -m unittest discover -s test -p '*.py'
