import keras
from keras.layers import merge, Lambda, Reshape, concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding1D, AveragePooling2D, Conv1D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential

from keras.optimizers import *
from keras import Input

class set_model():
    def __init__(self, data_base, model_name, pat_num, nb_layer, input_dim):
        self.data_base = data_base
        self.model_name = model_name
        self.pat_num = pat_num
        self.nb_layer = nb_layer
        self.input_dim = input_dim
        self.filter_size_1 = 3 
        self.strides_1 = 1 
        self.nb_filter = 16
        
    # concatenate when the same num_filter
    def identify_block(self, x, nb_filter, kernel_size = 3):
        # k1, k2, k3 = nb_filter
        k1, k2 = nb_filter
        out = Convolution2D(k1, (kernel_size, kernel_size), padding='same')(x)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Convolution2D(k2, (kernel_size, kernel_size), padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        # out =Conv2D(k3, (1, 1), padding='valid')(out)
        # out = BatchNormalization()(out)

        out = concatenate([out, x],axis=-1)
        out = Activation('relu')(out)
        return out
    
    # concatenate when different num_filter
    def conv_block(self, x, nb_filter, kernel_size = 3):
        # k1, k2, k3 = nb_filter
        k1, k2 = nb_filter
        out = Convolution2D(k1,(kernel_size, kernel_size), padding = 'same')(x)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Convolution2D(k2, (kernel_size,kernel_size), padding = 'same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        # out = Conv2D(k3, (1, 1), padding = 'valid')(out)
        # out = BatchNormalization()(out)

        x = Convolution2D(k2, (1, 1), padding = 'valid')(x)
        x = BatchNormalization()(x)

        out = concatenate([out,x],axis=-1)
        out = Activation('relu')(out)
        return out
    
	# build a MLP network 
    def MLP(self):
        inp = Input(shape=(self.input_dim,))
        
        out = Dense(input_dim=self.input_dim, units=self.nb_filter, activation='relu')(inp)
        
        for i in range(self.nb_layer):
            if (i+1)%4==1 and i!=0:
                self.nb_filter = self.nb_filter*2
                out = Dense(units=self.nb_filter, activation='relu')(out)
            else:
                out = Dense(units=self.nb_filter, activation='relu')(out)
        
        # model.add(Flatten())
        out = Dense(64)(out)
        out = Activation('relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(16)(out)
        out = Activation('relu')(out)
        out = Dense(1, activation='sigmoid')(out)
        
        model = Model(inp, out)
        model.summary()
        return model
    
    # build a VGG network
    def VGG(self, data_base):
        inp = Input(shape=(self.input_dim, 1))
        if data_base=='mimic':
            n_zeropadding = 0
            out = Conv1D(16, self.filter_size_1, strides=self.strides_1)(inp)
    
        elif data_base=='eicu':
            n_zeropadding = 1
            out = ZeroPadding1D(padding=n_zeropadding)(inp)
            out = Conv1D(16, self.filter_size_1, strides=self.strides_1)(out)
        
        # out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Reshape((int((self.input_dim + (2*n_zeropadding) - self.filter_size_1) / self.strides_1) + 1, 16, 1))(out)
        
        for i in range(self.nb_layer):
            if (i+1)%4==1:
                if i!=0:
                    self.nb_filter=self.nb_filter*2
                out = Convolution2D(self.nb_filter, (3, 3), padding='same')(out)
                out = Activation('relu')(out)
            else:
                out = Convolution2D(self.nb_filter, (3, 3), padding='same')(out)
                out = Activation('relu')(out)
        
        out = Flatten()(out)
        out = Dense(64)(out)
        out = Activation('relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(16)(out)
        out = Activation('relu')(out)
        
        out = Dense(1, activation='sigmoid')(out)
        
        model = Model(inp, out)
        model.summary()
        return model
    
    # build a Resnet network
    def Resnet(self, data_base):
        inp = Input(shape=(self.input_dim, 1))
        if data_base=='mimic':
            n_zeropadding = 0
            out = Conv1D(16, self.filter_size_1, strides=self.strides_1, activation='relu')(inp)
    
        elif data_base=='eicu':
            n_zeropadding = 1
            out = ZeroPadding1D(padding=n_zeropadding)(inp)
            out = Conv1D(16, self.filter_size_1, strides=self.strides_1)(out)
            
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        # print(out.shape)
        out = Reshape((int((self.input_dim + (2*n_zeropadding) - self.filter_size_1) / self.strides_1) + 1, 16, 1))(out)
        
        for i in range(int(self.nb_layer/2)):
            if (i+1)%2==1:
                if i!=0:
                    self.nb_filter=self.nb_filter*2
                out = self.conv_block(out, [self.nb_filter, self.nb_filter])
            
            else:
                out = self.identify_block(out, [self.nb_filter, self.nb_filter])

        #out = AveragePooling2D((3, 3))(out)
        
        out = Flatten()(out)
        out = Dense(64, activation='relu', name='Dense_1')(out)
        out = Dropout(0.5)(out)  
        out = Dense(16)(out)
        out = Activation('relu')(out)
        out = Dense(1, activation='sigmoid', name='Dense_2')(out)

        model = Model(inp, out)
        model.summary()
        # plot_model(model, to_file='model.png', show_shapes=True)
        
        return model
        
    def run(self):
        
        if self.model_name=='CNN':
            model = self.VGG(self.data_base)
            
        elif self.model_name=='Resnet':
            model = self.Resnet(self.data_base)
            
        elif self.model_name=='MLP':
            model = self.MLP()
            
        # compile a model
        model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='sgd')
        return model
