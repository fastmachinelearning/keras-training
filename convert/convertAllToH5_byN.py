import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--queue", type=str, default="8nh", help="LSFBATCH queue name"
    )
    parser.add_argument(
        "--resnet", action="store_true", help="224x224 image output for ResNet"
    )
    parser.add_argument(
        "-n", "--nevt", type=int, default=100, help="Number of events per file"
    )

    args = parser.parse_args()

    # prepare the list of files to run
    inFile = open("data/run_1evt.txt")
    linesIN = inFile.readlines()
    jobRanges = []
    for line in linesIN:
        values = line.split(" ")
        minVal = values[0]
        maxVal = values[1][:-1]
        fileName = "/eos/project/d/dshep/hls-fml/jetImage_1evt_%s_%s.h5" % (
            minVal,
            maxVal,
        )
        if os.path.isfile(fileName):
            continue
        jobRanges.append([minVal, maxVal])

    nToRun = len(jobRanges)
    nJob = nToRun // args.nevt

    for j in range(nJob + 1):
        script = open("jetImage/jetImage_%i.src" % j, "w")
        script.write(
            "source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-slc6-gcc62-opt/setup.sh\n"
        )
        if args.resnet:
            script.write(
                "python %s/script/jetImage_boundaries_ResNet.py %s %s\n"
                % (
                    os.getcwd(),
                    jobRanges[j * args.nevt][0],
                    jobRanges[(j + 1) * args.nevt][1],
                )
            )
        else:
            script.write(
                "python %s/script/jetImage_boundaries.py %s %s\n"
                % (
                    os.getcwd(),
                    jobRanges[j * args.nevt][0],
                    jobRanges[(j + 1) * args.nevt][1],
                )
            )
        script.close()
        os.system(
            "bsub -q %s -o jetImage/jetImage_%i.log -J jetImage_%i < jetImage/jetImage_%i.src "
            % (args.queue, j, j, j)
        )
        print("submitting job n. %i to the queue %s...\n" % (j, args.queue))
