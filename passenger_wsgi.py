import sys
import os

INTERP = os.path.join(os.environ['HOME'], 'path/to/venv/bin/python')
if sys.executable != INTERP:
    os.execl(INTERP, INTERP, *sys.argv)

from wsgi import app as application