import numpy as np
import cv2
import subprocess
import itertools
from multiprocessing import Pool




import sys
import os
import time
import matplotlib.pyplot as plt



f = subprocess.check_output(["ls"]).split()
files = []
#make list of files that contain ellipse data
for i in f:
    if "ellipseList.txt" in i:
        files.append(i)
print(files)

class Image:
	def __init__(self, filename, window_size):
		self.im = cv2.imread(filename,0)
		#self.im = cv2.resize(self.im,(0,0),fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
		self.mask = []
		self.mask_small = []
		self.windows = []
		self.windows_small = []
		self.scores = []
		self.scores_small = []
		self.cx = []
		self.cy = []
		self.decimation_factor = []
		self.imno = 0
		#self.slide = [-6,-4,-2,0,2,4,6]
		#self.slide = [-3,-2,-1,0,1,2,3]
                self.slide = [-2,0,2]
		self.window_size = window_size

	def ellipse(self, ellipse_info):
		ellipse_info = ellipse_info.split(" ")
		axes = [float(ellipse_info[0]),float(ellipse_info[1])]
		decim_fac = int(max(max(axes[0]*2/self.window_size,axes[1]*2/self.window_size),1))
		self.decimation_factor.append(decim_fac)

		#print "best decimation is %.2f and %.2f"%(axes[0]*2/32,axes[1]*2/32)
		theta = float(ellipse_info[2])
		self.cx.append(float(ellipse_info[3]))
		self.cy.append(float(ellipse_info[4]))
		#print "diameter is %0.2f"%(2*max(axes[0],axes[1]))
		y,x = np.ogrid[0:self.im.shape[0],0:self.im.shape[1]]
		mask = np.power(((x-self.cx[-1])*np.cos(theta) + (y-self.cy[-1])*np.sin(theta))/axes[0],2) + np.power(((x-self.cx[-1])*np.sin(theta) - (y-self.cy[-1])*np.cos(theta))/axes[1],2) <= 1
		self.mask.append(mask)
		#self.mask.append(mask[::2,::2])
		#self.cx[-1] /= 2
		#self.cy[-1] /= 2

	def ellipse_decim(self, ellipse_info):
		ellipse_info = ellipse_info.split(" ")
		axes = [float(ellipse_info[0])/2,float(ellipse_info[1])/2]
		print("best decimation is %.2f and %.2f"%(axes[0]*2/32,axes[1]*2/32))
		theta = float(ellipse_info[2])
		self.cx.append(float(ellipse_info[3])/2)
		self.cy.append(float(ellipse_info[4])/2)
		#print "diameter is %0.2f"%(2*max(axes[0],axes[1]))
		y,x = np.ogrid[0:self.im.shape[0],0:self.im.shape[1]]
		mask = np.power(((x-self.cx[-1])*np.cos(theta) + (y-self.cy[-1])*np.sin(theta))/axes[0],2) + np.power(((x-self.cx[-1])*np.sin(theta) - (y-self.cy[-1])*np.cos(theta))/axes[1],2) <= 1
		self.mask.append(mask)


	def get_score(self,mask,cx,cy,x,i,ellipse_size):
		s = self.window_size/2
		flag = False
		flag = flag or cy+x[0]-s < 0
		flag = flag or cx+x[0]-s < 0
		flag = flag or cy+x[1]+s+1 > mask.shape[0]
		flag = flag or cx+x[1]+s+1 > mask.shape[1]
		if flag == True:
			return -1.
		#intersect = np.sum(self.mask[i][cy+x[0]-16:cy+x[0]+17,cx+x[1]-16:cx+x[1]+17]).astype(float)
		#union = ellipse_size - intersect + (32*32)

		intersect = np.sum(mask[cy+x[0]-s:cy+x[0]+s+1,cx+x[1]-s:cx+x[1]+s+1]).astype(float)
		union = ellipse_size - intersect + (4*s*s)
		self.imno += 1

		#CHOOSE THE SCORE YOU WANT
		return np.float32(intersect/union)
		#return intersect/ellipse_size

	def get_random_window(self,image,mask,center):
		s = self.window_size/2
		rand_mask = mask[center[0]-s:center[0]+s+1,center[1]-s:center[1]+s+1]
		if rand_mask.size < (self.window_size**2) or np.sum(rand_mask) > 5:
			return None
		return image[center[0]-s:center[0]+s+1,center[1]-s:center[1]+s+1].astype(np.float32)

	def get_windows(self):
		s = self.window_size/2
		self.image_slides = []
		self.score_slides = []
		for i in xrange(len(self.mask)):
			image = cv2.resize(self.im,(0,0),fx=1./self.decimation_factor[i],fy=1./self.decimation_factor[i],interpolation=cv2.INTER_AREA)
			mask = cv2.resize(self.mask[i].astype(np.uint8),(0,0),fx=1./self.decimation_factor[i],fy=1./self.decimation_factor[i],interpolation=cv2.INTER_AREA).astype(bool)
			mask_size = np.sum(mask)
			cx = int(round(self.cx[i]/self.decimation_factor[i]))
			cy = int(round(self.cy[i]/self.decimation_factor[i]))
			self.score_slides.append(map(lambda x: self.get_score(mask,cx,cy,x,i,mask_size), itertools.product(self.slide,self.slide)))
			self.image_slides.append(map(lambda x: image[cy+x[0]-s:cy+x[0]+s+1,cx+x[1]-s:cx+x[1]+s+1].astype(np.float32), itertools.product(self.slide,self.slide)))
		
		#generate random images
		self.random_slides = []
		self.random_scores = []
		mask = np.zeros(self.im.shape)
		for i in xrange(len(self.mask)):
			mask = np.maximum(mask, self.mask[i].astype(int))
		mask = mask.astype(bool)
		rand = np.random.rand(self.imno,2)
		rand[:,0] *= self.im.shape[0]
		rand[:,1] *= self.im.shape[1]
		rand = rand.astype(int)
		iterate = 0
		goal = 2*self.imno
		while(self.imno < goal):
			try:
				randy = rand[iterate,0]
				randx = rand[iterate,1]
			except IndexError:
				rand = np.random.rand(self.imno,2)
				rand[:,0] *= self.im.shape[0]
				rand[:,1] *= self.im.shape[1]
				rand = rand.astype(int)
				iterate=0
				continue
			try:
				small = mask[randy-s:randy+s+1,randx-s:randx+s+1]
				#print "shape is %d %d"%(small.shape[0],small.shape[1])
				#print "val is %d"%np.sum(small)
			except IndexError:
				iterate+=1
				continue
			iterate+=1
			if small.size - (self.window_size**2) < 10:
				continue
			elif np.sum(small) > 10:
				continue
			self.random_slides.append(self.im[randy-s:randy+s+1,randx-s:randx+s+1].astype(np.float32))
			self.random_scores.append(np.float32(0))
			self.imno += 1
			#print "Adding random image"
			#print "%d left to go"%(goal-self.imno)

	def get_data(self):
		flatten = lambda l: [item for sublist in l for item in sublist]
		return flatten(self.image_slides)+self.random_slides, flatten(self.score_slides)+self.random_scores


def info(filename):
	with open(filename,"r") as f:
		slides = []
		scores = []
		while(True):
			try:
				imgpath = f.readline().split("\n")[0]+".jpg"
				if imgpath == ".jpg":
					return np.array(slides), np.array(scores)
				#print imgpath
				e = Image(imgpath,64)
				numfaces = f.readline().strip()
				#print numfaces
				print(numfaces)
				for i in xrange(int(numfaces)):
					ellipse_info = f.readline().split("\n")[0]
					#print ellipse_info
					e.ellipse(ellipse_info)
					#plt.imshow(e.im,cmap="gray",alpha=0.5)
					#plt.imshow(e.ellipse(ellipse_info),alpha=0.1,cmap="gray")
					#plt.show()
				e.get_windows()
				ims, im_scores = e.get_data()
				for i in xrange(len(ims)):
					slides.append(ims[i])
					scores.append(im_scores[i])
				#print
				#e.get_windows()
			except ValueError as a:
				#pass
				#    print e
				return
	#return

#info(files[0])
#exit()

pool = Pool(4)
a = np.array(pool.map(info,files))



images = np.concatenate(a[:,0]).tolist()
scores = np.concatenate(a[:,1]).tolist()

i=0
while(True):
	if i==len(images):
		break
	elif images[i].shape != (65,65):
		del images[i]
		del scores[i]
	else:
		i+=1

images = np.array(images)
scores = np.array(scores)




# images_flat = []
# scores_flat = []

# for i in xrange(len(images)):
# 	assert len(images[i]) == len(scores[i])
# 	for j in xrange(len(images[i])):
# 		print type(scores[i][j])
# 		images_flat.append(images[i][j])
# 		scores_flat.append(scores[i][j])

# images = np.array(images_flat)
# scores = np.array(scores_flat)

images = images[np.where(scores >= 0)]
scores = scores[np.where(scores >= 0)]
#scores_second = np.add(-1,scores)
#scores = np.concatenate((scores[:,np.newaxis],scores_second[:,np.newaxis]),axis=1)

#data = np.stack((images,scores[:,np.newaxis]),axis=1)
#np.random.shuffle(data)
#print(data.shape)


#plt.hist(np.ceil(scores),bins=50)
print images.shape
print scores.shape
np.save("x_train.npy",images.astype(np.float32))
np.save("y_train.npy",scores.astype(np.float32))
