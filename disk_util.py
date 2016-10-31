import os
import mmap
import json
import errno


def make_sure_path_exists(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise


def save_json(data, filename):
	with open(filename + '.json', 'wb') as f:
		json.dump(data, f, indent=2)


def unsure_folder_name(path):
	if path[-1] == '/':
		return
	return path + '/'


def countlines(filename):
	with open(filename, 'r+') as f:
		buf = mmap.mmap(f.fileno(), 0)
		lines = 0
		readline = buf.readline
		while readline():
			lines += 1
		return lines
