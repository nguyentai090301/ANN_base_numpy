import numpy as np
def normalize_data(data):
    mean = np.array([np.mean(data)]*len(data))
    std = np.array([np.std(data)]*len(data))
    data_nor = (np.array(data) - mean)/std
    return data_nor