# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:36:40 2019

@author: Sam Robinson
"""

#imports
from torchvision import models
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
import torch.nn.modules 
from torch.utils.tensorboard import SummaryWriter



writer = SummaryWriter()
#inisalize cuda cores and empty cache for use in training the mdoel
torch.cuda.empty_cache()
torch.cuda.init() 
#print the allocated memory to the screen for debugging
print(torch.cuda.max_memory_allocated())



#=============================================================================
#set the random seed to 100. the seed will allow us to get random images from the dataset such that the images and labels match. 
#seeding the datasets wil allow us to create a more accuarate 
SEED = 100
# these path pointed to places on the disk containing large datasets
# each containing about 3,000 images 

# the breast cancer dataset was a lot smaller and so a resnet50 was used  

img_path = 'E:\\training data\\BCImg\\'
lbl_path = 'E:\\training data\\BCLabels\\'
test_imgPath = 'E:\\training data\\BCVal\\'
test_imgPathlbl = 'E:\\training data\\BCLabelsVal\\'

"""
img_path = 'E:\\training data\\train\\'
lbl_path = 'E:\\training data\\labels\\'
test_imgPath = 'E:\\training data\\val\\'
test_imgPathlbl = 'E:\\training data\\vallabels\\'
"""
"""
img_path = 'E:\\training data\\BrainImages\\'
lbl_path = 'E:\\training data\\BrainLabels\\'
test_imgPath = 'E:\\training data\\BrainVal\\'
test_imgPathlbl = 'E:\\training data\\BrainvalLabels\\'
"""

#set up the diretory flows
#the data format must channel first as this is how pytorch takes tensors
train_imgs = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, data_format='channels_first')
test_imgs = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, data_format='channels_first')
train_labels = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, data_format='channels_first')
test_labels = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, data_format='channels_first')

#set up params for training images etc
train_img_gen = train_imgs.flow_from_directory(
    img_path,
    class_mode=None,
    batch_size=1,
    #the size of the image is 256 by 256. these must be matching for the labels and the inputs
    target_size=(256,256),
    seed=SEED)

train_label_gen = train_labels.flow_from_directory(
    lbl_path,
    class_mode=None,
    #the colour mode for the labels needs to be grayscale so that the network can acurately segment the images
    color_mode='grayscale',
    batch_size=1,
    target_size=(256,256),
    seed=SEED)

test_img_gen = test_imgs.flow_from_directory(
    test_imgPath,
    class_mode=None,
    batch_size=1,
    target_size=(256,256),
    seed=SEED)

test_label_gen = test_labels.flow_from_directory(
    test_imgPathlbl,
    class_mode=None,
    color_mode='grayscale',
    batch_size=1,
    target_size=(256,256),
    seed=SEED)

#create the model, fully connected resnet with 50 layers
#to make a fully connect resnet with 101 layers the code is as such: model = models.segmentation.fcn_resnet50(True) <- the true means this network 
# is trained on cats and dogs before and I use transfer learning to retrain the network. Sadly Pretrained resnets for 50 layers are not supported 
model = models.segmentation.fcn_resnet50(False)
torch.cuda.empty_cache()

#addiing more layers for upsampling and then adding logsoftmax activation function
#after experimenting with outputs and other acvtivation functins this was found to be the best
layers = []
#loop through all models layyers but the last layer 
for i in list(model.children())[:-1]:
    #for all items in the array model.childern
    for j in list(i):
        #append to the last layer is the type is the same as an empty string
        if type(j) == type(''):
            #append layers j to i
            layers.append(i[j])
        else:
            #else just append j 
            layers.append(j)

layers = layers[:-1]
"""
layers.append(torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)))
layers.append(torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)))
"""
#up sample layers allow the image to be up sammpled from  30 by 30, from the output of the resnet up 256 by 256
#by using KNN up sammpling
layers.append(torch.nn.UpsamplingNearest2d(scale_factor=2))
#match the new convsd layer we are adding to the previous layer by giving it 512 
layers.append(torch.nn.Conv2d(512 , 512, kernel_size=(3, 3),padding=(1, 1)))
layers.append(torch.nn.Conv2d(512 , 512, kernel_size=(3, 3),padding=(1, 1)))

layers.append(torch.nn.UpsamplingNearest2d(scale_factor=2))

layers.append(torch.nn.Conv2d(512 , 512, kernel_size=(3, 3),padding=(1, 1)))
layers.append(torch.nn.Conv2d(512, 256, kernel_size=(3, 3), padding =(1, 1)))

layers.append(torch.nn.UpsamplingNearest2d(scale_factor=2))

layers.append(torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)))
#16 output channles for the number of classes in each image
layers.append(torch.nn.Conv2d(256, 16, kernel_size=(3, 3), padding=(1, 1)))
#output activation function

layers.append(torch.nn.LogSoftmax(dim=1))

#make each layer Sequential
model = torch.nn.Sequential(*layers)
#model = nn.DataParallel(model)
#print entire netowrk for debugging
#print(list(model.children()))

#make new SummaryWriter for writing to logs folder
tb = SummaryWriter()
#put the model onto the GPU 
model.cuda()

#make a new optimizer alllowing the model to learn at certain rate, this chosen after a lot of expierment
# the learning rate the gradeiant deccent rate, if this number is set to any bigger than the gradienat is prone to 
#expldoing when getting very small

#the eps param and the amagrad can help stabilize the gradient descent so that the loss 
# does not explode when getting too small
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, eps=1e-3, amsgrad=True)

#criterion = torch.nn.modules.loss.MSELoss().cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
#this number will depend on the number images, it is the epoch number
#each dataset will be calcuated for 50 epochs (50 iterations over the data)_
for i in range(10000):    
        
        #set the model to training model
        model.train()
        #loop through the training set
        images = torch.Tensor(next(train_img_gen))
        #loop through the label set
        labels = torch.Tensor(next(train_label_gen))
            
        #make a long tensor out of the labels tensor. this has to be done as Cross Entropy Loss needs the target tensor to be
        #long tensors
        labels = labels.long()
        #squeeze the 0th element of the tensor
        #The squeeze() function below eliminates any dimension that has size 1
        labels = labels.squeeze(0)
        #put all images on the GPU
        images = images.cuda() 
        labels = labels.cuda()
        
        
        running_lossForTraining=0.0
        #set the grad to true for the omtimizer
        with torch.set_grad_enabled(True):
            #push the images throuh the network 
            y_pred = model.forward(images)
            #we need to set the gradients to zero before starting to do backpropragation because 
            #PyTorch accumulates the gradients on subsequent backward passes
            #zero the grad each time a new image is passed through the network
            optimizer.zero_grad()
            #compare the prediction to the labels 
            loss = criterion(y_pred, labels)
            #backwards propigate throuh the netowrk tweeking the weights of the nuerons to make the 
            #network better
            loss.backward()
            optimizer.step()
            #loss.item gets the loss number not the loss tensor
            running_lossForTraining = loss.item()
            #write the training loss to a scaler and label graph "training loss
            writer.add_scalar('training loss',
                              running_lossForTraining / 1000,
                              i* len(train_img_gen) + i
                              )  
        if(i % 100 == 0):
            
            
            y_pred = y_pred.cpu().detach().numpy().argmax(axis=1)
            plt.subplot(1, 3, 1)
            plt.imshow(images.cpu().detach().numpy()[0, 0, :, :])
            plt.subplot(1, 3, 2)
            plt.imshow(y_pred[0,:,:])
            plt.subplot(1, 3, 3)
            plt.imshow(labels.cpu().detach().numpy()[0, :, :])
            plt.show(block=True)
               
            print("epoch: ", i, "train loss:" , loss.item())   
        #this number is the epoch/50        
        if(i%68 == 0):
            #set the model to testing mode
            model.eval()
 
            images = torch.Tensor(next(test_img_gen))
            labels = torch.Tensor(next(test_label_gen))
                 
            
            labels = labels.long()
            labels = labels.squeeze(0)
        
            images = images.cuda() 
            labels = labels.cuda()
                
            running_lossForTest = 0.0
                
            y_pred = model.forward(images)
            loss = criterion(y_pred, labels)
            
            running_lossForTest = loss.item()               
            print("epoch: ", i, "test loss:" , loss.item())
            writer.add_scalar('testing loss', running_lossForTest / 1000,
                              i* len(test_img_gen) + i)
print("saving model...")           
#save the entire model along with all the weights so it can be used later segment 
#medical images or trained some more using transfer learning 
torch.save(model,"E:\\model\\50layerBreastModel1.0.2.pth" )  
print ("model saved.")
                            
            
#add the images to a grpah inside tensorboard after training. this is commented out as I feel like it is 
#not useful.
#images = torch.Tensor(next(train_img_gen))
#tb.add_graph(model.cuda(), images.cuda()) 

tb.close()   

#
# This is just plotting
#
"""
plt.subplot(2, 2, 1); plt.imshow(images.cpu().numpy()[0, 0, :, :])
plt.subplot(2, 2, 2); plt.imshow(images.cpu().numpy()[1, 0, :, :])
plt.subplot(2, 2, 3); plt.imshow(labels.cpu().numpy()[0, 0, :, :])
plt.subplot(2, 2, 4); plt.imshow(labels.cpu().numpy()[1, 0, :, :])
"""