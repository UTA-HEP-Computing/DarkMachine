import h5py from sklearn.preprocessing 
import RobustScaler 

def load_data(filename, len_input=80):
    # set input length len_input=80 load hdf5 dataset: data/AE_training_qcd.h5
    f_train=h5py.File(filename,'r')
    #f_validation=h5py.File('data/AE_validation_qcd.h5', 'r')
    x_train=f_train['table']
    #x_validation=f_validation['table']
    x_train.shape
    #x_validation.shape
    x_train=x_train[:,:len_input]
    #x_validation=x_validation[:,:200]
    scaler=RobustScaler().fit(x_train)
    x_train=scaler.transform(x_train)
    #x_validation=tf.transform(x_validation)
    f_train.close()
    return x_train
