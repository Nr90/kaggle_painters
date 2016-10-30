import csv
import random
import numpy as np
from tqdm import tqdm

np.random.seed(1)

datafile = open('data/features.csv', 'r')
trainfile = open('data/train.csv', 'a')
testfile = open('data/test.csv', 'a')
validationfile = open('data/validation.csv', 'a')
datasetreader = csv.reader(datafile)
datasetreader.next()
trainwriter = csv.writer(trainfile)
testwriter = csv.writer(testfile)
validationwriter = csv.writer(validationfile)
for line in tqdm(datasetreader):
	rand = random.randint(0, 10)
	if rand == 9:
		validationwriter.writerow(line)
	elif rand > 6:
		testwriter.writerow(line)
	else:
		trainwriter.writerow(line)
datafile.close()
trainfile.close()
testfile.close()
validationfile.close()
