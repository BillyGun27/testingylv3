from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from keras.layers import (Conv2D, GlobalAveragePooling2D, Input, Reshape,
                          ZeroPadding2D, UpSampling2D, Activation, Lambda, MaxPooling2D)
from keras.layers import add, Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot

def test():
    #model = load_model('data/qqwwee_yolo.h5')
    # return the constructed network architecture

   # yolo = Reshape((52, 52, 3, 85))(model.layers[-1].output)
    inputs = Input(shape=(416, 416, 3))
    x = Conv2D(64, (3, 3),
               padding='same',
               strides=1,
               activation='linear',
               kernel_regularizer=l2(5e-4))( inputs )
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)


    newmodel = Model( input= inputs , output=x )

    return newmodel

def custom():
    model = load_model('data/qqwwee_yolo.h5')

    # return the constructed network architecture
    yolo3 = Reshape((13, 13, 3, 85))(model.layers[-3].output)
    yolo2 = Reshape((26, 26, 3, 85))(model.layers[-2].output)
    yolo1 = Reshape((52, 52, 3, 85))(model.layers[-1].output)
    

    newmodel = Model( input= model.input , output=[yolo3,yolo2,yolo1] )

    return newmodel

if __name__ == '__main__':
    model = custom()
    print(model.summary())

    model = load_model('data/qqwwee_yolo.h5')
    #print(model.layers[-1].output)
    #print(model.layers[-2].output)


   # plot(model, to_file='data/debug_yolo.png', show_shapes=True)
   # print('Saved model plot to data/debug_yolo.png')
    #print(len(model.layers))