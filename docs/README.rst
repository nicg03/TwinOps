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

Publishing on Read the Docs
---------------------------
1. Create an account at https://readthedocs.org/ (or sign in with GitHub).

2. Click **Import a Project** and connect your Git provider (GitHub, GitLab, etc.)
   if not already connected.

3. **Import** the TwinOps repository. Read the Docs will detect the project and
   ask for a name (e.g. ``twinops``). The docs URL will be
   ``https://twinops.readthedocs.io/`` (or the name you chose).

4. The repository root must contain ``.readthedocs.yaml`` (already present).
   It configures:
   - Python 3.12 and Sphinx
   - ``pip install .[docs]`` so Sphinx and the theme are installed
   - Sphinx config: ``docs/conf.py``

5. Click **Build version** (or push to the default branch). The first build may
   take a few minutes (installs numpy, scipy, torch, sphinx, then builds the docs).

6. When the build succeeds, the documentation is available at
   ``https://<project-name>.readthedocs.io/``.

7. Optional: in **Admin → Advanced settings** you can set the default branch and
   enable **Build pull requests** for previews.

8. Add the Documentation link to your PyPI project page (Project links) so users
   can find it easily.
