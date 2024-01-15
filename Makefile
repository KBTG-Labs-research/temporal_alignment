PYTHON = venv/bin/python
PIP = venv/bin/pip

venv:
	python3.8 -m venv venv
	$(PIP) install -r requirements.txt -r dev-requirements.txt

test:
	$(PYTHON) -m pytest