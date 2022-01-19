from __future__ import absolute_import, division, print_function
import logging
import sys

def init_logging(level, color=False):
    """
    Setup logger.

    Parameters
    ----------
    level:        string, level of logging: DEBUG,INFO,WARNING,ERROR. (default: INFO).

    kwargs
    ------
    color:        bool, if true, enable colored output for bash output

    Notes
    -----
    for color see
        stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
        https://wiki.archlinux.org/index.php/Color_Bash_Prompt
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if level.upper() == 'INFO':
        level = logging.INFO
    elif level.upper() == 'DEBUG':
        level = logging.DEBUG
    elif level.upper() == 'WARNING':
        level = logging.WARNING
    elif level.upper() == 'ERROR':
        level = logging.ERROR


    if color:
        logging.basicConfig(level=level,stream = sys.stderr, format='\033[0;36m%(filename)10s:\033[0;35m%(lineno)4s\033[0;0m --- %(levelname)7s: %(message)s')
        logging.addLevelName( logging.DEBUG, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
        logging.addLevelName( logging.INFO, "\033[1;36m%s\033[1;0m" % logging.getLevelName(logging.INFO))
        logging.addLevelName( logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
        logging.addLevelName( logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    else:
        logging.basicConfig(level=level,stream = sys.stderr, format='%(filename)10s:%(lineno)4s --- %(levelname)7s: %(message)s')

    return
