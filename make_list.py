# usage: python make_list.py /home/nguyenmd/workspace/data/LIRIS/data_img liris.list
import sys
from os import listdir
from os.path import isfile, join, isdir, splitext


def readDir(path):
	return sorted([join(path, f) for f in listdir(path) if isdir(join(path, f))])

def readImageFiles(path):
	return sorted([join(path, f) for f in listdir(path) if isfile(join(path, f)) and splitext(f)[1] in (".jpg",)])

def makeImgList(path, outfile):
	with open(outfile, "w") as out:
		label_num = 0
		all_sub_dir = readDir(path)
		for sd in all_sub_dir:
			# all_img_file = readImageFiles(sd)
			# for imgfile in all_img_file:
			for i in range(100):
				out.write(sd + " " + str(label_num))
				out.write("\n")
			label_num += 1


if __name__=="__main__":
	makeImgList(sys.argv[1], sys.argv[2])
