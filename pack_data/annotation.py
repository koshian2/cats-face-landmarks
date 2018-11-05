import numpy as np
import os, glob, pickle

def convert_annotation_file_to_numpy(path):
    with open(path, "r") as fp:
        strs = fp.read().split()
    nums = [int(x) for x in strs]
    basepic = os.path.basename(path).replace(".cat", "")
    return [basepic, nums[0], np.array(nums[1:], dtype=np.int32).reshape(-1, 2)]

def convert_annotations():
    files = glob.glob("cats/*/*.cat")
    result = {}
    for file in files:
        data = convert_annotation_file_to_numpy(file)
        result[data[0]] = data
    with open("cat_annotation.dat", "wb") as fp:
        pickle.dump(result, fp)
    #print(result)

convert_annotations()
