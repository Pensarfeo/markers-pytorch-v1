import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'src')
lib_datasaver = osp.join('D:\Projects\libs\python\dataSaverDev')
add_path(lib_path) 

tests_path = osp.join(this_dir, 'tests')
add_path(tests_path) 
add_path(lib_datasaver)
