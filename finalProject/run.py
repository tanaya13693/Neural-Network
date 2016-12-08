from subprocess import call
import os
from time import time, sleep


def check(weights_seq, weights_cuda):
    relativeTolerance = 1e-6

    for i in xrange(len(weights_seq)):
        relativeError = weights_cuda[i] - weights_seq[i]
        # print "Comparing, cuda:", weights_cuda[i], "seq:", weights_seq[i], "Diff:", relativeError
        if (relativeError > relativeTolerance) or (relativeError < -relativeTolerance):
            print "Failed."
            return
    print "Passed."

if __name__ == "__main__":
    
    weights_cuda = list();
    weights_seq = list();

    print "\n\n\n\n*************GPU*************\n\n"

    os.chdir("cuda")

    print "\n\n***Cleaning***\n\n"
    call(["make clean"], shell=True)

    print "\n\n***Make***\n\n"
    call(["make"], shell=True)

    print "\n\n***Running***\n\n"
    call(["./run"], shell=True)

    with open("out.txt", "r") as f:
        for line in f.readlines()[1:]:
            weights_cuda.append(float(line.split(",")[2]))

    print "\n\n\n\n*************SEQUENTIAL*************\n\n"

    os.chdir("../sequential")

    print "\n\n***Cleaning***\n\n"
    call(["make clean"], shell=True)

    print "\n\n***Cleaning***\n\n"
    call(["make"], shell=True)
    
    print "\n\n***Running***\n\n"
    call(["./run"], shell=True)

    with open("out.txt", "r") as f:
        for line in f.readlines()[1:]:
            weights_seq.append(float(line.split(",")[2]))

    check(weights_seq, weights_cuda)
