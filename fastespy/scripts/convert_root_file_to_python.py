import ROOT
import argparse
import pickle
import bz2
import os
import time

"""
Stand alone script to convert root files of alpsIO class to python files on alpsdaq2
"""


if __name__ == "__main__":
    usage = "usage: %(prog)s -i infile -o outdir"
    description = "Convert root alpsIO files to python pickle files"
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-i', '--infile', required=True, help='Input .root file')
    parser.add_argument('-o', '--outdir', required=True, help='Output directory')
    args = parser.parse_args()

    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)

    alpsIOso = "/home/alpsdaq2/Misha/alpsIO/install/lib/libalpsIO.so"
    if not os.path.isfile(alpsIOso):
        raise IOError("alpsIO package does not seem to be installed in this location: {0:s}".format(alpsIOso))
    ROOT.gSystem.Load(alpsIOso)

    reader = ROOT.alpsIO.RootFileReader(args.infile)

    nevents = reader.getNEvents()
    print("found {0:n} triggers".format(nevents))

    res_ch0 = []
    res_ch1 = []

    outfile = os.path.join(args.outdir,
                           "{0:s}.pickle.bz2".format(os.path.basename(args.infile).split('.root')[0]))

    print("Writing data to file {0:s}".format(outfile))
    t1 = time.time()

    for i in range(nevents):
        records = reader.readRecords(i)

        for j, v in enumerate(dict(records).values()):

            if not j:
                res_ch0.append(dict(data=list(v.data),
                               samplingFreq=v.samplingFreq,
                               startTime=v.startTime,
                               trigger=v.trigger,
                               timeStamp=v.timeStamp,
                               channel=v.channel))
            else:
                res_ch1.append(dict(data=list(v.data),
                               samplingFreq=v.samplingFreq,
                               startTime=v.startTime,
                               trigger=v.trigger,
                               timeStamp=v.timeStamp,
                               channel=v.channel))
    
    with bz2.BZ2File(outfile, "w") as f:
        pickle.dump([res_ch0, res_ch1], f)

    print("It took {0:.2f} seconds".format(time.time() - t1))
