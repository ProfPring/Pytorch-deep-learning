# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:44:37 2020

@author: Sam
"""

#imports
import tkinter as tk
from tkinter import *
import torch  
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow
import sys
import os
#the root is the window to which all other elements are attached to 
root = tk.Tk()
label = tk.Label(root, text="if image has tummor then the network will outline it otherwise it will not outline anything")
#pack used to show label in gui
label.pack()

def loadbrainCancerImgs():
    img_path= 'newimg\\brain\\'
    #create a training image array to be laoded into the model
    #rescale changes any images to 255 by 255 
    train_imgs = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, data_format='channels_first')
    
    train_img_gen = train_imgs.flow_from_directory(
    img_path,
    class_mode=None,
    batch_size=1,
    target_size=(256,256))
     
    model = torch.load("model\\101layerBrainmodel1.0.2.pth")
    #put the model on the GPU
    model.cuda()
    #set the model to evaluation mode. sets dropout and batch normalization layers to evaluation mode 
    model.eval()
    #change the images into a tensor and iterate over the images
    images = torch.Tensor(next(train_img_gen))
    
    #put images on the GPU
    images = images.cuda()
    #set the gradients for params to trye for layers that need it.
    with torch.set_grad_enabled(False):
    
        #push the image through the network                 
        y_pred = model.forward(images) 
        #getting the max of the output tensor, which is softmax function of between 0 and 1
        #this will allow me to get a a number of which I can take the mean average of the values and then 
        #take the argmax which is given as: x∗=argmax/xf(x)
        
        #the arg max is: "the points, or elements, of the domain of 
        #some function at which the function values are maximized."
        brainmax = y_pred.cpu().detach().numpy().max(1)
        brainmax = np.mean(brainmax.argmax(1))
        print(brainmax)
        
        
        #the argmax gets the The maximum value along a given axis.
        #getting the argmax of the image this effectly the flatten the image into the most prelevent colours        
        y_pred = y_pred.cpu().detach().numpy().argmax(1)
        
        fig = plt.figure(figsize=(8,5))
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(wspace=0.025, hspace=0.05)
        
        ax1 = plt.subplot(1, 3, 1)
        #the detach() method constructs a new view on a tensor which is declared not to need gradients
        plt.imshow(images.cpu().detach().numpy()[0, 0, :, :])
        ax2 = plt.subplot(1, 3, 2)
        #the below code is making a label to sit under the image
        ax1.text(0.5,-0.1, "neural  network confidence", size=12, ha="center", 
         transform=ax1.transAxes)
        plt.imshow(y_pred[0,:,:])
        ax2.text(0.5,-0.1, brainmax*10000, size=12, ha="center", 
         transform=ax2.transAxes)
        #turn off the axsis numbers 
        ax1.axis('off')
        ax2.axis('off')
        
        plt.savefig('figs//figure1.png')
        #show figure as block. 
        plt.show(block=True)
            
    python = sys.executable
    os.execl(python, python, * sys.argv)
    #photo = tk.PhotoImage(file=file)
    
    #return so the main loop is not blocked up
    return file    

 

def loadNeckModelImgs():
    global label
    label['text'] = ''
    
    img_path= 'newimg\\neck\\'

    train_imgs = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, data_format='channels_first')
    
    train_img_gen = train_imgs.flow_from_directory(
    img_path,
    class_mode=None,
    batch_size=1,
    target_size=(256,256))
      
    model = torch.load("model\\101layerNeck1.0.4.pth")
    model.cuda()
    model.eval()
    
    images = torch.Tensor(next(train_img_gen))
    
    images = images.cuda()
    with torch.set_grad_enabled(False):
                 
        y_pred = model.forward(images) 
        print(y_pred)
    
        print(images.size())
        
        fig = plt.figure(figsize=(8,5))
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(wspace=0.025, hspace=0.05)
        
        
        y_pred = y_pred.cpu().detach().numpy().argmax(1)
        ax1 = plt.subplot(1, 3, 1)
        plt.imshow(images.cpu().detach().numpy()[0, 0, :, :])
        ax2 = plt.subplot(1, 3, 2)
        plt.imshow(y_pred[0,:,:])
        
        ax1.axis('off')
        ax2.axis('off')
        plt.savefig('figs//figure1.png')
        
        plt.show(block=True)

    python = sys.executable
    os.execl(python, python, * sys.argv)
    
    return model
        
def loadmodelbreast():
    global label
    label['text'] = ''
    
    img_path= 'newimg\\breast'

    train_imgs = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, data_format='channels_first')
    
    train_img_gen = train_imgs.flow_from_directory(
    img_path,
    class_mode=None,
    batch_size=1,
    target_size=(256,256))
       
    model = torch.load("model\\50layerBreastModel1.0.2.pth")
    model.cuda()
    model.eval()
    
    images = torch.Tensor(next(train_img_gen))
    
    images = images.cuda()
    
    with torch.set_grad_enabled(False):
                     
        y_pred = model.forward(images) 
        
        breastmax = y_pred.cpu().detach().numpy().max(1)
        breastmax = np.mean(breastmax.argmax(1))
        print(breastmax)
        
        print(y_pred)
        
        fig = plt.figure(figsize=(8,5))
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(wspace=0.025, hspace=0.05)
        
        print(images.size())
        print(y_pred.argmax(1))
        y_pred = y_pred.cpu().detach().numpy().argmax(1)
        ax1 = plt.subplot(1, 3, 1)
        plt.imshow(images.cpu().detach().numpy()[0, 0, :, :])
        ax2 = plt.subplot(1, 3, 2)
        
        ax1.text(0.5,-0.1, "neural network confidence", size=12, ha="center", 
         transform=ax1.transAxes)
        plt.imshow(y_pred[0,:,:])
        ax2.text(0.5,-0.1, breastmax*10000, size=12, ha="center", 
         transform=ax2.transAxes)
        
        plt.imshow(y_pred[0,:,:])
        
        ax1.axis('off')
        ax2.axis('off')
        
        plt.savefig('figs//figure1.png')
        plt.show(block=True)
        
    python = sys.executable
    os.execl(python, python, * sys.argv)
    return model    

# load saved figure into UI
file="figs//figure1.png"
print(file)
photo = tk.PhotoImage(file=file)
#make anew canvas on the root for the UI to diplay the in 
canvas = tk.Canvas(root, width = 500, height = 500)  
canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  
canvas.create_image(350,256, image=photo)
canvas.pack()
#.pack() command is used for displaying element to the GUI


#code for buttons, 
BrstC = tk.Button(root, text="segment breast cancer img", fg="black", command=loadmodelbreast)
BrstC.pack(side=tk.LEFT, padx=40, pady=10)
NECK = tk.Button(root, text="segment neck img", fg="black", command=loadNeckModelImgs)
NECK.pack(side=tk.LEFT, padx=60, pady=10)
BrnC = tk.Button(root, text="segment brain img", fg="black", command=loadbrainCancerImgs)
BrnC.pack(side=tk.LEFT, padx=40, pady=10)
#call the mainloop method to display the program
tk.mainloop()