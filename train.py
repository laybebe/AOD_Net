from keras import backend as K
import AODNet_model
import os
import glob
from math import ceil
from data_generate import generate
from preprocessing import getLabel_path
from preprocessing import ImageToArrayPreprocessor
from preprocessing import SimplePreprocessor
from keras.optimizers import SGD,Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0"
batch_sizes=32
epochs=20
current_path=os.path.abspath(os.path.dirname("__file__"))
label_directory=os.path.join(os.path.abspath(os.path.dirname("__file__")),"label_image")
data_path=glob.glob(os.path.join(current_path,"train_data/*.jpg"))
random.shuffle(data_path)
train_filenames=data_path[0:ceil(len(data_path)*0.8)]
val_filenames=data_path[ceil(len(data_path)*0.8):]
sp = SimplePreprocessor(640, 480)
iap = ImageToArrayPreprocessor()
data_product=generate(preprocessors=[sp, iap])
train_generator=data_product.data_generate(imagePaths=train_filenames,batch_sizes=batch_sizes)
validation_generator=data_product.data_generate(imagePaths=val_filenames,batch_sizes=batch_sizes)
# print(val_filenames)
# print(val_filenames[0],getLabel_path(val_filenames[0],label_directory))
os.environ["CUDA_VISIBLE_DEVICES"]="0"
model=AODNet_model.AOD_Net(input_height=640,input_width=480,input_channel=3,b=1).creat_AODNet()
opt=Adam(lr=0.001)
model.compile(loss="mean_squared_error", optimizer=opt)
print("[INFO] training network...")
early_stopping =EarlyStopping(monitor='val_loss', patience=2)
model_name = 'AODNet'
tensorboard_vis = TensorBoard(log_dir='./logs', histogram_freq=0, 
                                  batch_size=batch_sizes, write_graph=True, 
                                  write_grads=True, write_images=True, 
                                  embeddings_freq=0, embeddings_layer_names=None, 
                                  embeddings_metadata=None)
H=model.fit_generator(generator = train_generator,
                      steps_per_epoch = ceil(len(train_filenames)/batch_sizes),
                      epochs = epochs,
                      validation_data = validation_generator,
                      validation_steps = ceil(len(val_filenames)/batch_sizes),
                    callbacks=[tensorboard_vis])

model.save('./model_file/{}.h5'.format(model_name))
model.save_weights('./model_file/{}_weights.h5'.format(model_name))
print("Model saved under {}.h5".format(model_name))
print("Weights also saved separately under {}_weights.h5".format(model_name))
# model.summary()
print("[INFO] Draw a chart ...")
# list all data in history
print(H.history.keys())
plt.plot(np.arange(0, epochs),H.history['loss'],label="train_loss")
plt.plot(np.arange(0, epochs),H.history['val_loss'],label="val_loss")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('lossVSepoch.png')
plt.show()