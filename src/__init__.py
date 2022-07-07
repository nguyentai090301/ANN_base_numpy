name = 'ANN_base_numpy'
import os
import pandas as pd
from NormalizeData import normalize_data
from ANN import ANN
if __name__ == '__main__':
    data_dir = './data'
    train_path =  os.path.join(data_dir , 'train_record.csv')
    test_path =  os.path.join(data_dir , 'test_record.csv')
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)
    # Normalize data on train and test dataset
    data_train['platelets'] = normalize_data.normalize_data(data_train['platelets'].tolist())
    data_train['creatinine_phosphokinase'] = normalize_data.normalize_data(data_train['creatinine_phosphokinase'].tolist())
    data_test['platelets'] = normalize_data.normalize_data(data_test['platelets'].tolist())
    data_test['creatinine_phosphokinase'] = normalize_data.normalize_data(data_test['creatinine_phosphokinase'].tolist())
    # Get feature into array
    feature = data_train.columns[:-1]
    X_train = data_train.loc[:, feature].values
    X_test = data_test.loc[:, feature].values

    y_train = data_train['DEATH_EVENT'].tolist()
    y_test = data_test['DEATH_EVENT'].tolist()
    model = ANN.NeuralNetwork()
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.compute_accuracy(y_test, y_pred)
    print(y_pred)
    print(score)
     
    
    
    
    