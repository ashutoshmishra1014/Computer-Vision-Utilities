from PIL import Image
import os
from collections import defaultdict
import scipy.misc
import numpy as np

# folder = "/home/ashutosh/Documents/Playground/_ve_/mg_projects/handwriting_challenge/code/drn/new_dataset_training/gt"
# new_folder= "/home/ashutosh/Documents/Playground/_ve_/mg_projects/handwriting_challenge/code/drn/new_dataset_training/new"

folder = "/home/ashutosh/Documents/Playground/_ve_/mg_projects/handwriting_challenge/code/drn/final_dataset/gt"
# new_folder = "/home/ashutosh/Documents/Playground/_ve_/mg_projects/handwriting_challenge/dataset/manual_annotation/Liwicki-extra-dataset-1-extra_saurabh/gt_poc_voc_style"
palette = {(0,   0,   0) : 0 , 
         (255,  0, 0) : 1 ,
         (0,  0,  255) : 2
          }

# palette = {(128, 64, 128): 0, 
# 			(244, 35, 232): 255,
# 			(70, 70, 70): 0
# 			}

# 0: background
# 1: comment
# 2: textline

#prints all the unique pixel color values
def get_color():
	for _file in os.listdir(folder):
		im = Image.open(folder+'/'+_file)
		by_color = defaultdict(int)
		for pixel in im.getdata():
			by_color[pixel] += 1
		print(_file+":")
		print(by_color)

#changes the RGB image into a grayscale image as per the pallete mentioned above
def change_color():
	for _file in os.listdir(folder):
		picture = np.array(Image.open(folder+"/"+_file)) #Can be many different formats.
		print(_file)
		new_picture = convert_from_color_segmentation(picture)
		scipy.misc.imsave(folder+"/"+_file, new_picture)
		print("done for this file")


def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d

def change_2_classes_color():
	for _file in os.listdir(folder):
		print(_file)
		picture = np.array(Image.open(folder+"/"+_file)) #Can be many different formats.
		picture[picture==2]=0
		scipy.misc.imsave(folder+"/"+_file, picture)
		print("done for this file")


change_color()
