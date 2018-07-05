import numpy as np 
import dlib
import cv2

def resizeImg(img, size=1000):
	h, w = img.shape[:2]
	ratio = None
	if h > w :
		if h > size :
			ratio = size / h
		else :
			ratio = 1.0
	else :
		if w > size :
			ratio = size / w
		else :
			ratio = 1.0
	new_img = cv2.resize(img,None,fx=ratio, fy=ratio, interpolation = cv2.INTER_CUBIC)
	return new_img

def getBoxFromRect(rect):
	#top left
	x = rect.left()
	y = rect.top()
	#height width
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def getPosFromRect(rect):
	return (rect.left(), rect.top(), rect.right(), rect.bottom())

def shapeToNp(shape):
	result = [(shape.part(i).x, shape.part(i).y) for i in range(5)]
	result = np.array(result, dtype='int')
	return result

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

image = cv2.imread('faces/5.jpg')
image = resizeImg(image)
# bnw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
bnw = image

rects = detector(image, 1)
print('Faces =', len(rects))

# faces = dlib.full_object_detections()

for (i, rect) in enumerate(rects):
	shape = predictor(image, rect)
	aligned = dlib.get_face_chip(image, shape, padding=0.5)
	shape = shapeToNp(shape)
	(left, top, right, bottom) = getPosFromRect(rect)
	
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	cv2.putText(image, "Face #{}".format(i + 1), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	for(x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	
	cv2.imshow('detect {}'.format(i+1), image)
	cv2.imshow('align {}'.format(i+1), aligned)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	continue
	
# cv2.imshow('output', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()