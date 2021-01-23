import pandas as pd
import numpy as np
import personal_cleaner as pc
import other_cleaner as oc
from sklearn import preprocessing
import category_encoders as ce


def clean_data(personal_train, personal_valid, other_train, other_valid):
    # Cistenie datasatov personal
    personal_train, personal_valid = pc.clean_personal(personal_train, personal_valid)
    
    # Cistenie datasetov other
    other_train, other_valid = oc.clean_data(other_train, other_valid)
       
    # Zlucenie datasetov personal a other pre train a valid
    data_train = pd.merge(personal_train, other_train
                          , how='left'
                          , left_on=['name','address', 'city', 'state']
                          , right_on = ['name','address', 'city', 'state'])
    data_valid = pd.merge(personal_valid, other_valid
                          , how='left'
                          , left_on=['name','address', 'city', 'state']
                          , right_on =['name','address', 'city', 'state'])
    
    # Vymazanie atributov name a address v upravenych datasetoch - pre ďalšiu fázu spracovania nie sú potrebné
    del data_train['name']
    del data_train['address']
    del data_valid['name']
    del data_valid['address']
    
    # Ak po spojani ostali zaznamy, ku ktorym neboli hodnoty - vymazanie
    data_train = data_train.dropna(subset=['class'])
    data_valid = data_valid.dropna(subset=['class'])
    
    # Zmena kategorickych atributov na numericke - binarne
    data_train['sex'] = data_train.apply(lambda row: 1 if 'Male'in row['sex'] else 0, axis=1)
    data_valid['sex'] = data_valid.apply(lambda row: 1 if 'Male'in row['sex'] else 0, axis=1)
    
    # Zmena kategorickych atributov na numericke - hash
    
    # city
    city_encoder = preprocessing.LabelEncoder()
    city_list = np.unique(np.concatenate((data_train['city'], data_valid['city']), axis=None))
    city_encoder.fit(city_list)
    data_train['city'] = city_encoder.transform(data_train['city'])
    data_valid['city'] = city_encoder.transform(data_valid['city'])
    
    # state
    state_encoder = preprocessing.LabelEncoder()
    state_list = np.unique(np.concatenate((data_train['state'], data_valid['state']), axis=None))
    state_encoder.fit(state_list)
    data_train['state'] = state_encoder.transform(data_train['state'])
    data_valid['state'] = state_encoder.transform(data_valid['state'])
    
    # country
    country_encoder = preprocessing.LabelEncoder()
    a = np.concatenate((data_train['country'], data_valid['country']), axis=None)
    country_list = np.unique(a.astype(str))
    country_encoder.fit(country_list)
    data_train['country'] = country_encoder.transform(data_train['country'])
    data_valid['country'] = country_encoder.transform(data_valid['country'])
    
    # date
    data_train['date_of_birth'] = data_train['date_of_birth'].str.replace("-","").astype(int)
    data_valid['date_of_birth'] = data_valid['date_of_birth'].str.replace("-","").astype(int)
    
    # Zmena kategorickych atributov na numericke - one-hot
    encoder = ce.OneHotEncoder()
    encoder.fit(data_train, data_train['mean_oxygen'])
    data_train = encoder.transform(data_train)
    data_valid = encoder.transform(data_valid)
    
    return data_train, data_valid
