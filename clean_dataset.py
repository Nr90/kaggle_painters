import csv
import keras.preprocessing.image as image_utils

infile = open('data/manifest.csv', 'r')
outfile = open('data/manifest_clean.csv', 'a')
csvreader = csv.reader(infile)
csvreader.next()
csvwriter = csv.writer(outfile)
csvwriter.writerow(['filenames', 'artist'])
idx = 1
for line in csvreader:
	filepath = 'data/images/' + line[0]
	try:
		image = image_utils.load_img(filepath, target_size=(224, 224))
		image = image_utils.img_to_array(image)
		csvwriter.writerow(line)
		print 'Created clean entry %s' % idx
		idx += 1
	except Exception as e:
		print 'Exception: %s' % e
infile.close()
outfile.close()
