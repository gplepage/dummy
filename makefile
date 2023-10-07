# PYTHONLIB = python installation directory if use 'make install'
# make sure this is part of PYTHONPATH (in .bashrc, eg)
PIP = python -m pip
PYTHON = python
PYTHONVERSION = python`python -c 'import platform; print(platform.python_version())'`
VERSION = `python -c 'import dummy; print(dummy.__version__)'`

SRCFILES := $(shell ls setup.py pyproject.toml src/dummy/*.py)
DOCFILES := $(shell ls doc/*.rst doc/conf.py)

install-user :
	$(PIP) install . --user --no-cache-dir

install install-sys :
	$(PIP) install . --no-cache-dir

uninstall :			# mostly works (may leave some empty directories)
	$(PIP) uninstall dummy

update:
	make uninstall install


.PHONY : doc

doc/html/index.html : $(SRCFILES) $(DOCFILES) setup.cfg
	sphinx-build -b html doc/ doc/html

doc-html doc:
	make doc/html/index.html

clear-doc:
	rm  -rf doc/html

.PHONY : tests

tests:
	python -m unittest discover

coverage:
	pytest --cov-report term-missing --cov=dummy tests/

sdist:          # source distribution
	$(PYTHON) -m build --sdist


upload-twine: $(CYTHONFILES)
	twine upload dist/dummy-$(VERSION).tar.gz

upload-git: $(CYTHONFILES)
	echo  "version $(VERSION)"
	make doc-html
	git diff --exit-code
	git diff --cached --exit-code
	git push origin master

tag-git:
	echo  "version $(VERSION)"
	git tag -a v$(VERSION) -m "version $(VERSION)"
	git push origin v$(VERSION)

test-download:
	-$(PIP) uninstall dummy
	$(PIP) install dummy --no-cache-dir
