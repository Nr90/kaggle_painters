import csv
import random
import numpy as np
from tqdm import tqdm

np.random.seed(1)

datafile = open('data/manifest_clean.csv', 'r')
trainfile = open('data/train.csv', 'a')
testfile = open('data/test.csv', 'a')
validationfile = open('data/validation.csv', 'a')
datasetreader = csv.reader(datafile)
datasetreader.next()
trainwriter = csv.writer(trainfile)
trainwriter.writerow(['filename', 'artist'])
testwriter = csv.writer(testfile)
testwriter.writerow(['filename', 'artist'])
validationwriter = csv.writer(validationfile)
validationwriter.writerow(['filename', 'artist'])
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
