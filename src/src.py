from typing import Any
import tensorflow as tf
import numpy as np
from abc import abstractmethod
from keras import layers
from keras.models import Model


def get_UNet3D(input_shape=(25,25,49,1)):
    
    input = layers.Input(shape=input_shape)

    s = input

    s = layers.Cropping3D(cropping=((1, 0), (1, 0),(1, 0)), data_format=None)(s) # this is the added step


    c1 = layers.Conv3D(32, (3, 3, 3), padding='same') (s)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)
    c1 = layers.Conv3D(32, (3, 3, 3), padding='same') (c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)
    p1 = layers.MaxPool3D((2, 2, 2)) (c1)

    c2 = layers.Conv3D(64, (3, 3, 3), padding='same') (p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation("relu")(c2)
    c2 = layers.Conv3D(64, (3, 3, 3), padding='same') (c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation("relu")(c2)
    p2 = layers.MaxPool3D((2, 2, 2)) (c2)

    c3 = layers.Conv3D(128, (3, 3, 3), padding='same') (p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation("relu")(c3)
    c3 = layers.Conv3D(128, (3, 3, 3), padding='same') (c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation("relu")(c3)
    p3 = layers.MaxPool3D((2, 2, 2)) (c3)

    u7 = layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same') (p3)
    

    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv3D(128, (3, 3, 3), padding='same') (u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation("relu")(c7)
    c7 = layers.Conv3D(128, (3, 3, 3), padding='same') (c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation("relu")(c7)

    u8 = layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same') (c7)
    


    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv3D(64, (3, 3, 3), padding='same') (u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation("relu")(c8)
    c8 = layers.Conv3D(64, (3, 3, 3), padding='same') (c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation("relu")(c8)

    u9 = layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same') (c8)


    
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv3D(32, (3, 3, 3), padding='same') (u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation("relu")(c9)
    c9 = layers.Conv3D(32, (3, 3, 3), padding='same') (c9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation("relu")(c9)

    c10 = layers.ZeroPadding3D(padding=((1, 0),(1, 0),(1, 0)))(c9)


    c10 = layers.ZeroPadding3D(padding=((1, 1),(1, 1),(1, 1)))(c10)
    c10 = layers.Conv3D(64, (3, 3, 3), padding='valid') (c10)
    c10 = layers.BatchNormalization()(c10)
    c10 = layers.Activation("relu")(c10)


    outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid') (c10)


    return Model(input, outputs)

def get_CV_UNet(input_shape=(25,25,49,2)):
    return get_UNet3D(input_shape=input_shape)

def get_UNet2D(input_shape=(25,25,49,1)):
    input = layers.Input(shape=input_shape)

    s = input

    s = layers.Cropping3D(cropping=((1, 0), (1, 0),(1, 0)), data_format=None)(s) # this is the added step


    c1 = layers.Conv3D(32, (3, 3, 1), padding='same') (s)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)
    c1 = layers.Conv3D(32, (3, 3, 1), padding='same') (c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)
    p1 = layers.MaxPool3D((2, 2, 2)) (c1)

    c2 = layers.Conv3D(64, (3, 3, 1), padding='same') (p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation("relu")(c2)
    c2 = layers.Conv3D(64, (3, 3, 1), padding='same') (c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation("relu")(c2)
    p2 = layers.MaxPool3D((2, 2, 2)) (c2)

    c3 = layers.Conv3D(128, (3, 3, 1), padding='same') (p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation("relu")(c3)
    c3 = layers.Conv3D(128, (3, 3, 1), padding='same') (c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation("relu")(c3)
    p3 = layers.MaxPool3D((2, 2, 2)) (c3)

    u7 = layers.Conv3DTranspose(128, (2, 2, 1), strides=(2, 2, 2), padding='same') (p3)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv3D(128, (3, 3, 1), padding='same') (u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation("relu")(c7)
    c7 = layers.Conv3D(128, (3, 3, 1), padding='same') (c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation("relu")(c7)

    u8 = layers.Conv3DTranspose(64, (2, 2, 1), strides=(2, 2, 2), padding='same') (c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv3D(64, (3, 3, 1), padding='same') (u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation("relu")(c8)
    c8 = layers.Conv3D(64, (3, 3, 1), padding='same') (c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation("relu")(c8)

    u9 = layers.Conv3DTranspose(32, (2, 2, 1), strides=(2, 2, 2), padding='same') (c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv3D(32, (3, 3, 1), padding='same') (u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation("relu")(c9)
    c9 = layers.Conv3D(32, (3, 3, 1), padding='same') (c9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation("relu")(c9)

    c10 = layers.ZeroPadding3D(padding=((1, 0),(1, 0),(1, 0)))(c9)


    c10 = layers.ZeroPadding3D(padding=((1, 1),(1, 1),(0, 0)))(c10)
    c10 = layers.Conv3D(64, (3, 3, 1), padding='valid') (c10)
    c10 = layers.BatchNormalization()(c10)
    c10 = layers.Activation("relu")(c10)

    outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid') (c10)

    return Model(input, outputs)

def get_ResNet(input_shape=(25, 25, 49, 1)):

    input = layers.Input(input_shape)

    s = input

    c1 = layers.Conv3D(64, (3, 3, 3), padding='same') (s)
    c1 = layers.Activation("relu")(c1)

    c1 = layers.Conv3D(64, (3, 3, 3), padding='same') (c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)

    c1 = layers.Conv3D(64, (3, 3, 3), padding='same') (c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)

    c1 = layers.Conv3D(64, (3, 3, 3), padding='same') (c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)

    c1 = layers.Conv3D(64, (3, 3, 3), padding='same') (c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)

    c1 = layers.Conv3D(64, (3, 3, 3), padding='same') (c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)

    c1 = layers.Conv3D(64, (3, 3, 3), padding='same') (c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)

    c1 = layers.Conv3D(64, (3, 3, 3), padding='same') (c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)

    c1 = layers.Conv3D(64, (3, 3, 3), padding='same') (c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)

    inter = layers.Conv3D(1, (3, 3, 3), padding='same') (c1)

    outputs = layers.add([inter, input])

    return Model(input, outputs)

class _ApproachWrapper():
    def __init__(self, DNN, A=None) -> None:
        
        if type(DNN) ==str:
            self.DNN = tf.keras.models.load_model(DNN, compile=False)
        else: 
            self.DNN=DNN
        
        self.A=A
    
    @abstractmethod
    def predict(self,y):
        pass

class Deep2S(_ApproachWrapper):
    def __init__(self, DNN, A=None) -> None:
        super().__init__(DNN, A)
        self.A_conj=np.conjugate(A)

    def _normalize(self,intermediate_reconst,normalize_on_batch=False):
        
        mag=np.abs(intermediate_reconst)

        if normalize_on_batch:
            intermediate_reconst/=mag.max()
        else:
            intermediate_reconst/=np.max(mag,axis=(-1,-2,-3),keepdims=True)
        return intermediate_reconst
    
    def predict(self, y):
        AHy = np.einsum('ktrxyz,Nktr->Nxyz',self.A_conj,y)
        AHy_mag_normalized = self._normalize(np.abs(AHy),normalize_on_batch=True)
        return np.squeeze(self.DNN.predict(AHy_mag_normalized[...,None]))        

class Deep2SP(_ApproachWrapper):
    def __init__(self, DNN, A=None) -> None:
        super().__init__(DNN, A)

    def predict(self, y):
        y_ri = np.concatenate((np.real(y[...,None]),np.imag(y[...,None])),axis=-1)
        return np.squeeze(self.DNN.predict(y_ri))        

class CV_Deep2S(_ApproachWrapper):
    def __init__(self, DNN, A=None) -> None:
        super().__init__(DNN, A)
    
    def predict(self, y):
        AHy = np.einsum('ktrxyz,Nktr->Nxyz',self.A_conj,y)
        AHy_normalized = self._normalize(AHy)
        AHy_normalized_ri = np.concatenate((np.real(AHy_normalized[...,None]),np.imag(AHy_normalized[...,None])),axis=-1)
        return np.squeeze(self.DNN.predict(AHy_normalized_ri))        

class DeepDI(_ApproachWrapper):
    def __init__(self, DNN, A=None) -> None:
        super().__init__(DNN, A)

    def predict(self, y):
        y_flatten = np.concatenate([np.real(y).reshape((y.shape[0],y.shape[1]*y.shape[2]*y.shape[3],1)),
                        np.imag(y).reshape(((y.shape[0],y.shape[1]*y.shape[2]*y.shape[3],1)))],axis=-1)
        return np.squeeze(self.DNN.predict(y_flatten))        

