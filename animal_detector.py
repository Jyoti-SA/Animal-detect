# USAGE
# python cat_detector.py --image images/cat_01.jpg

# import the necessary packages

#from _future_ import print_function
from firebase import firebase


import argparse
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-c", "--cascade",
	default="haarcascade_frontalface.xml",
	help="path to animal detector haar cascade")
args = vars(ap.parse_args())

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the detector Haar cascade, then detect  faces
# in the input image
detector = cv2.CascadeClassifier(args["cascade"])
rects = detector.detectMultiScale(gray, scaleFactor=1.3,
	minNeighbors=10, minSize=(75, 75))
count=0
# loop over the  faces and draw a rectangle surrounding each
for (i, (x, y, w, h)) in enumerate(rects):
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.putText(image, "animal {}".format(i + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
	count =len(rects)
	#str count= str(rects.shape[0]), (0,image.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1
	cv2.putText(image, "Number of animals detected: " +str(rects.shape[0]), (0,image.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1 )
firebase = firebase.FirebaseApplication('https://sd14cs043-160006.firebaseio.com/')
data = {'animals': count}
result = firebase.put('https://sd14cs043-160006.firebaseio.com','/count', data)
print (result)
        # show the detected  faces
cv2.imshow("Faces", image)
cv2.waitKey(0)
