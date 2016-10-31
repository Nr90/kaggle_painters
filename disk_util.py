import mmap


def countlines(filename):
	with open(filename, 'r+') as f:
		buf = mmap.mmap(f.fileno(), 0)
		lines = 0
		readline = buf.readline
		while readline():
			lines += 1
		return lines
