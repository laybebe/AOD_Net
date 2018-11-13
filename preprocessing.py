from keras.preprocessing.image import img_to_array
# import the necessary packages
import cv2
import os
'''keras和tensorflow默认的data_format 是 NHWC，caffe 默认的data_format是NCHW'''
class ImageToArrayPreprocessor:

    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the the Keras utility function that correctly rearranges
        # the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)


class SimplePreprocessor:
    def __init__(self, height, width, inter=cv2.INTER_AREA):   
    # store the target image width, height, and interpolation
    # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # ratio
        return cv2.resize(image, (self.width, self.height),
            interpolation=self.inter)
def getLabel_path(TrainingData_path,label_directory):
    image_name=os.path.basename(TrainingData_path).split("_")[0]+"_"+os.path.basename(TrainingData_path).split("_")[1]+".jpg"
    Label_path=os.path.join(label_directory,image_name)
    return Label_path
    
    
    
    