import cv2
import numpy as np
from keras.models import load_model
import sys
import itertools

model = load_model(sys.argv[1])

cap = cv2.VideoCapture(0)
ret, image = cap.read()
image = cv2.resize(image ,(0,0),fx=0.25, fy=0.25)
x_range = image.shape[0]-65
y_range = image.shape[1]-65


split = list(itertools.product(xrange(0,x_range,20),xrange(0,y_range,20)))
s = np.array(split)
print s.shape

def sliding_window(image):
    x_range = image.shape[0]-65
    y_range = image.shape[1]-65
    return np.array(map(lambda x: image[x[0]:x[0]+65,x[1]:x[1]+65], split))

while(True):
    ret, frame = cap.read()

    image = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(0,0),fx=0.25, fy=0.25)
    gray = np.repeat(gray[:,:,np.newaxis],3,axis=2)

    data = sliding_window(gray)

    print np.amax(model.predict(data))
    #location = np.where(model.predict(data) >= .5)
    #for i in s[location[0]]:
    #    cv2.rectangle(image,tuple(i*4),tuple((i*4)+65*4),(255,255,255),5)

    
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
