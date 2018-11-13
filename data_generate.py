import numpy as np
import cv2
import os
from preprocessing import getLabel_path
label_directory=os.path.join(os.path.abspath(os.path.dirname("__file__")),"label_image")
class generate(object):
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors
        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def data_generate(self,imagePaths=[],batch_sizes=32,shuffle=True):
        current=0
        while True:
            batch_data=[]
            batch_labels=[]
            if current>=len(imagePaths):
                current=0
                if shuffle:
                    np.random.shuffle(imagePaths)
            batch_filenames=imagePaths[current:current+batch_sizes]
            for file in batch_filenames:
                label_path=getLabel_path(file,label_directory)
                image = cv2.imread(file)
                label_img=cv2.imread(label_path)
                if self.preprocessors is not None:
                    # loop over the preprocessors and apply each to
                    # the image
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                        label_img = p.preprocess(label_img)
                    batch_data.append(image)
                    batch_labels.append(label_img)
            yield (np.array(batch_data).astype("float")/255.0, np.array(batch_labels).astype("float")/255.0)


    def data_generate_test(self, imagePaths=[], batch_sizes=32, shuffle=True):
        current = 0
        while True:
            batch_data = []
            if current >= len(imagePaths):
                current = 0
                if shuffle:
                    np.random.shuffle(imagePaths)
            batch_filenames = imagePaths[current:current + batch_sizes]
            for file in batch_filenames:
                image = cv2.imread(file)
                if self.preprocessors is not None:
                    # loop over the preprocessors and apply each to
                    # the image
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                    batch_data.append(image)
            yield (np.array(batch_data).astype("float") / 255.0)




