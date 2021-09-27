#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import cv2 
from tqdm import tqdm
import pandas as pd

class PixelMapper(object): 
    """ Create an object for converting pixels to geographic coordinates, using four points with known locations which form a quadrilteral in both planes Parameters ---------- pixel_array : (4,2) shape numpy array The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left pixels of the known region lonlat_array : (4,2) shape numpy array The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left pixels of the known region """ 
    
    def __init__(self, pixel_array, lonlat_array): 
        assert pixel_array.shape==(4,2), "Need (4,2) input array" 
        assert lonlat_array.shape==(4,2), "Need (4,2) input array" 
        
        self.M = cv2.getPerspectiveTransform(np.float32(pixel_array),np.float32(lonlat_array)) 
        self.invM = cv2.getPerspectiveTransform(np.float32(lonlat_array),np.float32(pixel_array)) 
        
    def pixel_to_lonlat(self, pixel): 
        """ Convert a set of pixel coordinates to lon-lat coordinates 
        
        Parameters 
        ---------- 
        pixel : (N,2) numpy array or (x,y) tuple The (x,y) pixel coordinates to be converted 
        
        Returns 
        ------- 
        (N,2) numpy array The corresponding (lon, lat) coordinates """ 
        
        if type(pixel) != np.ndarray: 
            pixel = np.array(pixel).reshape(1,2) 
        
        assert pixel.shape[1]==2, "Need (N,2) input array" 
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0],1))], axis=1) 
        lonlat = np.dot(self.M,pixel.T) 
        
        return (lonlat[:2,:]/lonlat[2,:]).T 
        
    def lonlat_to_pixel(self, lonlat): 
        """ Convert a set of lon-lat coordinates to pixel coordinates 
        
        Parameters 
        ---------- 
        lonlat : (N,2) numpy array or (x,y) tuple The (lon,lat) coordinates to be converted 
        
        Returns 
        ------- 
        (N,2) numpy array The corresponding (x, y) pixel coordinates """ 
        
        if type(lonlat) != np.ndarray: 
            lonlat = np.array(lonlat).reshape(1,2) 
        
        assert lonlat.shape[1]==2, "Need (N,2) input array" 
        lonlat = np.concatenate([lonlat, np.ones((lonlat.shape[0],1))], axis=1) 
        pixel = np.dot(self.invM,lonlat.T) 
        
        return (pixel[:2,:]/pixel[2,:]).T
    
# final new court: Use this one!
#[height, width]
#[,top left, top right, bottom right, bottom left]

quad_coords = { 
                "pixel": np.array([[273, 317],[273,696], [650,958], [650,70]]), #match video
                "lonlat": np.array([[2 ,  2],[358,2],[360,542], [2, 543]]) #floor pic
              }

dd =  PixelMapper(quad_coords["pixel"], quad_coords['lonlat'])


#Load Bounding Box Data and sort out the person you're looking for
people_data = pd.read_csv(r"C:\Users\h00ns\Tracking\organized_labels_yh_sy.csv")

people_data = people_data[people_data['item'] == 1190]
people_data = people_data[people_data['frame'] > 7348]

#create list of coordinates to mark on court
people = []
for idx in tqdm(range(len(people_data))):
    x = people_data['x_coordinate'].iloc[idx]
    y = people_data['y_coordinate'].iloc[idx]
    
    people.append([y,x])
    
    

#map coordinates onto image of squash court floor
final = people

pink = [255,0,255]
img = cv2.imread(r"C:\Users\h00ns\Tracking\court_floor_final.jpg", 1)

transformed = []


for point in final: #point = [height, width]
    height = point[0]
    width = point[1]
    dot = dd.pixel_to_lonlat([height, width])
    int_point = [int(dot[0][1]), int(dot[0][0])] 
    transformed.append(int_point)

print((len(transformed)), "[height, width]")

#original

for int_point in tqdm(transformed):
    try:
        for x in range(-5,5):
            img[int_point[0]+x, int_point[1]] = pink
            img[int_point[0], int_point[1]+x] = pink

    except IndexError:
        
        print("error", int_point)
        pass
    
#show player movement map in separate window
cv2.namedWindow('image2',cv2.WINDOW_NORMAL)
cv2.imshow('image2', img)
cv2.waitKey(0)

