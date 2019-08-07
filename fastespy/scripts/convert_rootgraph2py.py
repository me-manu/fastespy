from __future__ import absolute_import, division, print_function
from fastespy.rootdata import readgraph
import argparse

if __name__ == '__main__':
    usage = "usage: %(prog)s -d directory [-s suffix]"
    description = "Convert root graphs to numpy files"
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-d', '--directory', required=True, help='Directory with npz data files')
    parser.add_argument('-s', '--suffix', help='suffix for data files', default='tes2')
    args = parser.parse_args()

    t, v, tin, vin = readgraph(args.directory, prefix=args.suffix)
