 
import cv2
filename = 'new_img2.jpg'
W = 1500.
oriimg = cv2.imread(filename)
height, width, depth = oriimg.shape
print(depth)
if width>height :
    imgScale = W/width
else :
    imgScale = W/height
newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
newimg = cv2.resize(oriimg,(int(newX),int(newY)))
cv2.imshow("Show by CV2",newimg)
cv2.waitKey(0)
cv2.imwrite("resizeimg.jpg",newimg)
