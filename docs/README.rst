Building the documentation
==========================

Requirements
------------
Install the documentation dependencies::

  pip install -e ".[docs]"

or::

  pip install sphinx sphinx-rtd-theme

Build
-----
From the **repository root** (TwinOps/)::

  sphinx-build -b html docs docs/_build

Or with the Makefile::

  make -C docs html

The generated HTML is in ``docs/_build/html/`` (or ``docs/_build/`` depending on Sphinx version).
Open ``docs/_build/html/index.html`` or ``docs/_build/index.html`` in your browser.

Development (auto-reload)
-------------------------
To rebuild the docs on file save::

  pip install sphinx-autobuild
  sphinx-autobuild docs docs/_build
