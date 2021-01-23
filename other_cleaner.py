import re
import pandas as pd
import numpy as np
import remove_outliers
import imputer_median
import imputer_average


# Funkcia transformuje atribut pregnant - uprava formatu, doplnenie NaN a prevod na numericky atribut
def transform_pregnant(data):
    data['pregnant'] = data['pregnant'].replace(to_replace ='f', value = 'F', regex = True)
    data['pregnant'] = data['pregnant'].replace(to_replace ="FALSE", value = 'F', regex = True)
    data['pregnant'] = data['pregnant'].replace(to_replace ='t', value = 'T', regex = True)
    data['pregnant'] = data['pregnant'].replace(to_replace ='TRUE', value = 'T', regex = True)
    
    data.loc[data.pregnant.isnull(), 'pregnant'] = 'F'
    
    data['pregnant'] = data.apply(lambda row: 1 if 'T'in row['pregnant'] else 0, axis=1)
    return data


# Funkcia transformuje atribut education_num - doplnenie NaN, prevod na int, vymazanie atributu education
def transform_education_num(data):
    data = data.reset_index(drop=True)
    new_values = []
    index = 0
    education_num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    education_list = ['Preschool', '1st_4th', '5th_6th', '7th_8th', '9th', '10th', '11th', '12th', 'HS_grad', 'Some_college',
                 'Assoc_voc', 'Assoc_acdm', 'Bachelors', 'Masters', 'Prof_school', 'Doctorate']
    i = 0
    while i < len(data.education_num):
        if (data.education_num.isnull()[i] == True):
            if(data.education.isnull()[i] == False):
                index = education_list.index(data.education[i])
                value = education_num_list[index]
                new_values.append(value)
        i += 1
    data.loc[data.education_num.isnull(), 'education_num'] = new_values
    
    data['education_num'] = data.education_num.astype(int)
    
    del data['education']
    return data


# Funkcia rozdeli hodnotu z atributu personal info do novych atributov
# Format:  'Craft-repair|United-States\r\r\nMarried-civ-spouse -- Self-emp-not-inc|Asian-Pac-Islander'
def split_personal_info(text):
    occupation, tmp1, race = text.split('|')
    country, tmp1 = tmp1.split('\r\r\n')
    marital, workclass = tmp1.split(' -- ')
    return pd.Series([occupation, country, marital, workclass, race])
    

# Funkcia rozdeli hodnotu z atributu address do novych atributov    
def split_address(text):
    address, tmp = text.replace('\r', '').split('\n')
    #APO, which stands for Army Post Office
    tmp = tmp.replace('APO', 'APO,')
    #FPO, which stands for Fleet Post Office
    tmp = tmp.replace('FPO', 'FPO,')
    #DPO, which stands for Diplomatic Post Office
    tmp = tmp.replace('DPO', 'DPO,')
    city, state = tmp.split(', ', 1)
    return pd.Series([address, city, state])


# Funkcia v hodnotach vybranych atributov vymaze biele medzery a vymeni '-' za '_'
def remove_white_spaces(data, col):
    for i in col:
        data[i] = data[i].replace(to_replace ='[\s]', value = '', regex = True)
        data[i] = data[i].replace(to_replace ='-', value = '_', regex = True)
    return data


# Funkcia doplni za nezname hodnoty hodnotu 'Unknown'
def mark_missing_values(data, col):
    for i in col:
        data[i] = data[i].replace(to_replace ='\?+', value = 'Unknown', regex = True)
        data[i] = data[i].replace(to_replace ='nan', value = 'Unknown', regex = True)
    data.loc[data.relationship.isnull(), 'relationship'] = 'Unknown'
    data['country'] = data.country.astype(str)
    data['country'] = data['country'].replace(to_replace = np.nan, value = 'Unknown', regex = True)
    data.loc[data.country.isnull(), 'country'] = 'Unknown'
    return data


# Funkcia zmeni datovy typ atributu age z float na int
def convert_hours_per_week_type(data):
    data['hours_per_week'] = data.hours_per_week.astype(int)
    return data


# Funkcia doplni NaN hodnoty by default a transformuje atribut na numericky - binarne
def transform_income(data):
    data.loc[data.income.isnull(), 'income'] = '<=50K'
    data['income'] = data.apply(lambda row: 1 if '>50K'in row['income'] else 0, axis=1)
    return data


# Funkcia normalizuje hodnoty atributu predelenim odlahlych hodnot identifikovanym parametrom 100
def norm_mean_glucose(data):
    mask_bigger = (data['mean_glucose'] > 175 )
    mask_smaller = (data['mean_glucose'] < 0 )
    data.loc[mask_bigger, 'mean_glucose'] = data.loc[mask_bigger, 'mean_glucose'] / 100
    data.loc[mask_smaller, 'mean_glucose'] = data.loc[mask_smaller, 'mean_glucose'] / (-100)
    return data


# Normalizacia hodnot atributu pomocou logaritmickej funkcie
def norm_by_log(X):
    offset = 0
    if(X.min() < 0):
        offset = abs(X.min()) + 1
    if(X.min() == 0):
        offset = 1
    X = np.log(X+offset)
    return X


#  Normalizacia hodnot atributu pomocou odmocniny
def norm_by_sqrt(X):
    offset = 0
    if(X.min() < 0):
        offset = abs(X.min())
    X = np.sqrt(X+offset)
    return X


# Funkcia vykoná jednoduche transformácie nad vstupným datasetom
def preprocessor(data):
    pd.set_option('mode.chained_assignment', None)
    # Fix formatovania nazvov atributov
    data.columns = data.columns.str.replace('-', '_')

    # Odstranenie riadkov s NaN hodnotou v personal_info
    data = data.dropna(subset=['personal_info'])

    # Rozdelenie atributu personal_info(deleted) na atributy occupation/country/marital/workclass/race
    data[['occupation', 'country', 'marital', 'workclass', 'race']] = data.personal_info.apply(split_personal_info)
    del data['personal_info']
    
    # Fix formatu hodnot v sltpcoch income, education, relationship - odstranenie bielych medzier a zmena z '-' na '_'
    data = remove_white_spaces(data, ['income', 'education', 'relationship'])
    
    # Oznacenie chybajucich hodnot na Unknown
    data = mark_missing_values(data, ['occupation', 'country', 'marital', 'workclass', 'race'])

    # Rozdelenie atributu address na atributy address/city/state
    data[['address', 'city', 'state']] = data.address.apply(split_address)

    # Deduplikacia - zlucenie duplicitnych zaznamov do jedneho
    data = data.groupby('name').first().reset_index()

    # Odstranenie riadkov s NaN hodnotou v class
    data = data.dropna(subset=['class'])

    # Fix formatu hodnot v sltpci pregnant
    data = transform_pregnant(data)

    # Doplnenie hodnot atributu education_num na zaklade hodnoty atributu education
    data = transform_education_num(data)
    
    # Doplnenie hodnot atributu income a prevod na numericky-binarny atribut
    data = transform_income(data)
    
    # Nahradenie NaN hodnot v capital loss, capital gain
    data.loc[data.capital_loss.isnull(), 'capital_loss'] = 0
    data.loc[data.capital_gain.isnull(), 'capital_gain'] = 0
    
    return data


def clean_data(data_train, data_valid):
    data_train = preprocessor(data_train)
    data_valid = preprocessor(data_valid)
    
    # Doplnenie NaN hodnot atributu hours_per_week na zaklade medianu. Prevod na int.
    hours_imp = imputer_median.imputer()
    hours_imp.fit(data_train['hours_per_week'])
    data_train['hours_per_week'] = hours_imp.transform(data_train['hours_per_week'])
    data_train = convert_hours_per_week_type(data_train)
    data_valid['hours_per_week'] = hours_imp.transform(data_valid['hours_per_week'])
    data_valid = convert_hours_per_week_type(data_valid)
    
    # Doplnenie NaN hodnot atributu hours_per_week na zaklade medianu.
    fnlwgt_imp = imputer_median.imputer()
    fnlwgt_imp.fit(data_train['fnlwgt'])
    data_train['fnlwgt'] = fnlwgt_imp.transform(data_train['fnlwgt'])
    data_valid['fnlwgt'] = fnlwgt_imp.transform(data_valid['fnlwgt'])
    
    # Normalizacia atriutu std_glucose a doplnenie hodnot medianom
    rsg = remove_outliers.remove()
    rsg.fit(data_train['std_glucose'])
    data_train['std_glucose'] = rsg.transform(data_train['std_glucose'])
    data_valid['std_glucose'] = rsg.transform(data_valid['std_glucose'])
    
    imp_std_g = imputer_median.imputer()
    imp_std_g.fit(data_train['std_glucose'])
    data_train['std_glucose'] = imp_std_g.transform(data_train['std_glucose'])
    data_valid['std_glucose'] = imp_std_g.transform(data_valid['std_glucose'])
    
    # Normalizacia atributu mean_glucose a doplnenie hodnot priemerom
    data_train = norm_mean_glucose(data_train)
    data_valid = norm_mean_glucose(data_valid)
    
    imp_mean_g = imputer_average.imputer()
    imp_mean_g.fit(data_train['mean_glucose'])
    data_train['mean_glucose'] = imp_mean_g.transform(data_train['mean_glucose'])
    data_valid['mean_glucose'] = imp_mean_g.transform(data_valid['mean_glucose'])
    
    # Normalizacia atributu skewness_glucose a doplnenie hodnot priemerom
    data_train['skewness_glucose'] = norm_by_log(data_train['skewness_glucose'])
    data_valid['skewness_glucose'] = norm_by_log(data_valid['skewness_glucose'])
    
    imp_skew_g = imputer_average.imputer()
    imp_skew_g.fit(data_train['skewness_glucose'])
    data_train['skewness_glucose'] = imp_skew_g.transform(data_train['skewness_glucose'])
    data_valid['skewness_glucose'] = imp_skew_g.transform(data_valid['skewness_glucose'])
    
    # Normalizacia atributu kurtosis_glucose a doplnenie hodnot medianom
    data_train['kurtosis_glucose'] = norm_by_sqrt(data_train['kurtosis_glucose'])
    data_valid['kurtosis_glucose'] = norm_by_sqrt(data_valid['kurtosis_glucose'])
    
    imp_kurt_g = imputer_median.imputer()
    imp_kurt_g.fit(data_train['kurtosis_glucose'])
    data_train['kurtosis_glucose'] = imp_kurt_g.transform(data_train['kurtosis_glucose'])
    data_valid['kurtosis_glucose'] = imp_kurt_g.transform(data_valid['kurtosis_glucose'])   
    
    # Doplnenie NaN hodnot pre atribut skewness_oxygen - medianom
    data_train['skewness_oxygen'] = norm_by_log(data_train['skewness_oxygen'])
    data_valid['skewness_oxygen'] = norm_by_log(data_valid['skewness_oxygen'])
    
    imp_skew_o = imputer_median.imputer()
    imp_skew_o.fit(data_train['skewness_oxygen'])
    data_train['skewness_oxygen'] = imp_skew_o.transform(data_train['skewness_oxygen'])
    data_valid['skewness_oxygen'] = imp_skew_o.transform(data_valid['skewness_oxygen'])
    
    # Doplnenie NaN hodnot pre atribut std_oxygen - medianom
    data_train['std_oxygen'] = norm_by_log(data_train['std_oxygen'])
    data_valid['std_oxygen'] = norm_by_log(data_valid['std_oxygen'])
    
    imp_std_o = imputer_median.imputer()
    imp_std_o.fit(data_train['std_oxygen'])
    data_train['std_oxygen'] = imp_std_o.transform(data_train['std_oxygen'])
    data_valid['std_oxygen'] = imp_std_o.transform(data_valid['std_oxygen'])
    
    # Doplnenie NaN hodnot pre atribut mean_oxygen - priemerom
    data_train['mean_oxygen'] = norm_by_log(data_train['mean_oxygen'])
    data_valid['mean_oxygen'] = norm_by_log(data_valid['mean_oxygen'])
    
    imp_mean_o = imputer_average.imputer()
    imp_mean_o.fit(data_train['mean_oxygen'])
    data_train['mean_oxygen'] = imp_mean_o.transform(data_train['mean_oxygen'])
    data_valid['mean_oxygen'] = imp_mean_o.transform(data_valid['mean_oxygen'])
    
    # Doplnenie NaN hodnot pre atribut kurtosis_oxygen - priemerom
    data_train['kurtosis_oxygen'] = norm_by_sqrt(data_train['kurtosis_oxygen'])
    data_valid['kurtosis_oxygen'] = norm_by_sqrt(data_valid['kurtosis_oxygen'])
    
    rko = remove_outliers.remove()
    rko.fit(data_train['kurtosis_oxygen'])
    data_train['kurtosis_oxygen'] = rko.transform(data_train['kurtosis_oxygen'])
    data_valid['kurtosis_oxygen'] = rko.transform(data_valid['kurtosis_oxygen'])
    
    imp_kurt_o = imputer_average.imputer()
    imp_kurt_o.fit(data_train['kurtosis_oxygen'])
    data_train['kurtosis_oxygen'] = imp_kurt_o.transform(data_train['kurtosis_oxygen'])
    data_valid['kurtosis_oxygen'] = imp_kurt_o.transform(data_valid['kurtosis_oxygen'])
    
    return data_train, data_valid
