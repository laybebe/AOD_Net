#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Multiply,Subtract
from keras.regularizers import l2,l1
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda
# from keras import backend as K
from keras.activations import relu
import keras as keras
from keras.models import Model
class AOD_Net(object):
    """
    This is an implementation of AOD-Net : All-in-One Network for Dehazing on Python3, keras.
    The model can removal hazy, smoke or even water impurities.
    """
    def __init__(self,input_height=None,input_width=None,input_channel=None,b=None):
        self.heigt=input_height
        self.width=input_width
        self.channel=input_channel
        self.b=b                  #the constant bias
    def creat_AODNet(self,l2_regularization=0.0001):
        l2_reg = l2_regularization
        inputs=Input(shape=(self.heigt,self.width,self.channel))
        conv1 = Conv2D(3, (1, 1), kernel_initializer='random_normal', activation='relu',padding="same" ,
                       kernel_regularizer=l2(l2_reg),name="conv1")(inputs)

        conv2 = Conv2D(3, (3, 3), kernel_initializer='random_normal', activation='relu',padding="same" ,
                       kernel_regularizer=l2(l2_reg),name="conv2")(conv1)

        concat1=concatenate([conv1,conv2],axis=-1,name="concat1")

        conv3 = Conv2D(3, (5, 5), kernel_initializer='random_normal', activation='relu', padding="same",
                       kernel_regularizer=l2(l2_reg), name="conv3")(concat1)

        concat2 = concatenate([conv2, conv3], axis=-1,name="concat2")

        conv4 = Conv2D(3, (5, 5), kernel_initializer='random_normal', activation='relu', padding="same",
                       kernel_regularizer=l2(l2_reg), name="conv4")(concat2)

        concat3 = concatenate([conv1,conv2, conv3,conv4], axis=-1, name="concat3")

        K_x = Conv2D(3, (5, 5), kernel_initializer='random_normal', activation='relu', padding="same",
                       kernel_regularizer=l2(l2_reg), name="K_x")(concat3)

        """
          formulation：
          I(x) = J(x)*t(x) + A*(1 − t(x))
          J(x)=K(x)*I(x)-K(x)+b
          where :
          J(x)is the scene radiance (i.e., the ideal “clean image”)
          I(x) is observed hazy image
          A denotes the global atmosphericlight
          t(x) is the transmission matrix 
        """

        mul=Multiply(name="mul")([K_x,inputs])
        sub=Subtract(name="sub")([mul,K_x])
        add_b=Lambda(lambda x:1+x,name="add_b")(sub)
        output=Lambda(lambda x:relu(x),name="output")(add_b)
        model=Model(inputs=inputs,outputs=output)
        return model








