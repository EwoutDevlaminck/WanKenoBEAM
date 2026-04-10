# WKBeam setup

# This variable should point to the desired python interpreter
# Both python 2.7 and python 3.x are supported.

# python interpreter of choice
PYINT = /home/devlamin/WanKenoBEAM/venv/bin/python3

# corresponding f2py (from the WanKenoBeam venv — numpy 1.26.4)
# Using an absolute path so the build is independent of which Python venv
# is currently activated in the shell.
FTOPY = /home/devlamin/WanKenoBEAM/venv/bin/f2py
