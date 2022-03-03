import pandas as pd
from sklearn import preprocessing
import numpy as np

def sample_rct():
    df = pd.read_csv("actg175.csv", header=0, index_col=0)
    
    cd4_baseline = df['cd40']
    cd4_20 = df['cd420']
    
    outcome = (cd4_20 - cd4_baseline).values
    cov_cont = df[['age', 'wtkg', 'cd40', 'karnof', 'cd80']]
    cov_cont_norm = preprocessing.scale(cov_cont)
    cov_bin = df[['gender', 'homo', 'race', 'drugs', 'symptom', 'str2', 'hemo']]
    cov_bin_val = cov_bin.values 
    t = df[['arms']].values
    
    data = np.concatenate((cov_cont_norm, cov_bin_val, t.reshape(-1,1), outcome.reshape(-1,1)), axis=1)
    data.shape
    
    #Only focus on one arm (0=zidovudine, 1=zidovudine and didanosine, 2=zidovudine and zalcitabine,3=didanosine)
    t_1 = 2
    t_0 = 0
    t_ind = (t == t_0) + (t == t_1)
    
    data_rct = data[t_ind.flatten()]
    #change treatment sign to 1
    data_rct[:,-2] = np.where(data_rct[:,-2] == 2, 1, 0)
    
    return [data_rct[:,:-2], data_rct[:,-1],  data_rct[:,-2]]



