# Data Exploration in Python
Allen Downey

Supporting code for a video series on best practices for exploratory data analysis.


## Software setup
This repository contains the software I'll demonstrate in the videos.  If you have a Git client installed, you should be able to download the code by running:

    git clone https://github.com/AllenDowney/DataExplorationInPython.git

It should create a directory named DataExplorationInPython.  Otherwise you can download it as a zip file from https://github.com/AllenDowney/DataExplorationInPython/archive/master.zip

To run the code, you need Python 2 or 3 with IPython, NumPy, SciPy, and matplotlib.  I highly recommend installing Anaconda, which is a Python distribution that includes everything you need to run my code.  It is easy to install on Windows, Mac, and Linux; and because it does a user-level install, it will not interfere with other Python installations.  Information about Anaconda is at http://continuum.io/downloads.

To test your environment, start IPython:

    cd DataExplorationInPython
    ipython notebook

A browser window should appear with a list of files in this directory.  If IPython didn't create a browser window for you, you can do it yourself.  When you started the notebook, you should have seen a message like this:

    2015-04-02 15:44:33.254 [NotebookApp] Using existing profile dir: u'/home/downey/.ipython/profile_default'
    2015-04-02 15:44:33.267 [NotebookApp] Using MathJax from CDN: http://cdn.mathjax.org/mathjax/latest/MathJax.js
    2015-04-02 15:44:33.280 [NotebookApp] Serving notebooks from local directory:     /home/downey/DataExplorationInPython
    2015-04-02 15:44:33.280 [NotebookApp] 0 active kernels 
    2015-04-02 15:44:33.280 [NotebookApp] The IPython Notebook is running at: http://localhost:8888/
    2015-04-02 15:44:33.280 [NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

This is the startup message from the IPython server.  It includes the URL you can use to connect to the server.  Launch a browser and paste in this URL.

Click on `effect_size.ipynb`, which is one of the notebooks I'll demonstrate.  It should open a new tab and load the notebook.

Execute the first few cells by pressing Shift-Enter a few times, or from the Cell menu select "Run All".

If you don't get any error messages, you have everything you need.  Otherwise the error message should indicate what you are missing.
