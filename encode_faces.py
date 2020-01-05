# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle

# import the necessary packages
import argparse
import cv2
import os
import pickle
from imutils import paths
import face_recognition

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset, then initialize
# out data list (which we'll soon populate)
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	print(imagePath)
	image = cv2.imread(imagePath)
	# HEIGHT, WIDTH, DEPTH = image.shape
	# if(HEIGHT > 1500 or WIDTH > 1500):
	# 	if(WIDTH > HEIGHT):
	# 		SCALE_FACTOR = 1500/WIDTH
	# 	else:
	# 		SCALE_FACTOR = 1500/HEIGHT

	# 	NEW_X, NEW_Y = image.shape[1]*SCALE_FACTOR, image.shape[0]*SCALE_FACTOR
	# 	NEW_IMAGE = cv2.resize(image, (int(NEW_X), int(NEW_Y)))
	# else:
	# 	NEW_IMAGE = image

	NEW_IMAGE = image
	rgb = cv2.cvtColor(NEW_IMAGE, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,number_of_times_to_upsample=2
		model=args["detection_method"])

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# build a dictionary of the image path, bounding box location,
	# and facial encodings for the current image
	d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
		for (box, enc) in zip(boxes, encodings)]
	data.extend(d)

# dump the facial encodings data to disk
print("[INFO] serializing encodings...")
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()