
from utils import *
from model import set_model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy import interp
from sklearn.metrics import roc_curve, auc

seed = 222
np.random.seed(seed)

np.set_printoptions(threshold=np.inf)
os.environ['KERAS_BACKEND'] = 'tensorflow'

'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
'''

class Train():
    def __init__(self, args):
        self.nb_epoch = 250
        self.data_base = args.data_base     # MIMIC, eICU
        self.model_name = args.model_name   # CNN, Resnet, MLP
        self.nb_layer = args.nb_layer
        
        if self.data_base=='mimic':
            self.batch_size = 16
            self.input_dim = 16
            self.path = "./data_set/mimic_processed_1D.npz"
            
        elif self.data_base=='eicu':
            self.batch_size = 512
            self.input_dim = 15
            self.path = "./data_set/eicu_processed_1D.npz"


    def cross_validation(self, x, y, pat_num):
        tprs = []
        aucs = []  
        mean_fpr = np.linspace(0, 1, 100)
        
        if self.model_name=='MLP':
            x = x.reshape(x.shape[0], self.input_dim)
        else:
            x = x.reshape(x.shape[0], self.input_dim, 1)
        
        i = 1
        # 5-fold cross-validation
        kFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)    
        for train, test in kFold.split(x, y):
            print('This is the', i, 'fold')
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min', baseline=None, restore_best_weights=False)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='min', min_delta=0.000001)
            m = set_model(self.data_base, self.model_name, pat_num, self.nb_layer, self.input_dim)
            model = m.run()
            model.fit(x[train], y[train], batch_size=self.batch_size, epochs=self.nb_epoch, validation_data=(x[test], y[test]), verbose=1, shuffle=True, 
                    callbacks=[roc_callback(training_data=[x[train], y[train]], validation_data=[x[test], y[test]]), reduce_lr, early_stop])
            probas_1 = model.predict(x[test])
            print(probas_1)
           
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_1)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC', lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#696969', alpha=.4, label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        
    def run(self):
        sys.stdout = Logger(str(self.data_base)+"_train_record_"+str(self.model_name)+str(self.nb_layer)+".txt")
        cvscores = []
        start = time.clock()
    
        data = load_data(self.path)
        pat_num = data["arr_0"]
        # print('pat_num is', pat_num)
        data0 = data["arr_1"]
        # print('data is', data0)
        label = data["arr_2"]
        # print('label is ', label)  

        self.cross_validation(data0, label, pat_num)
        # print ('the training is ok!')
        end = time.clock()
        print('Time cost is:', end-start)
        
        print('finished!')
    
        
        
        
   
    
    
    

    

    

   