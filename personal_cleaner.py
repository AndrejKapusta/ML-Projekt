import re
import pandas as pd
from datetime import datetime

# Funckcia vymaze biele znaky z hodnôt atribútu sex
def fix_sex(data):
    data['sex'] = data['sex'].replace(to_replace ='[\s]', value = '', regex = True)
    return data


# Funkcia upraví/ujednotí formát dátumu narodenia na formát yyyy-mm-dd
def fix_dates(data):
    date_format = '-'
    data['date_of_birth'] = data['date_of_birth'].replace(to_replace ='00:00:00', value = '', regex = True)
    data['date_of_birth'] = data['date_of_birth'].replace(to_replace ='00 00 00', value = '', regex = True)
    data['date_of_birth'] = data['date_of_birth'].replace(to_replace ='/', value = '-', regex = True)
    
    dates = data.date_of_birth[:]
    
    i = 0
    while i < len(dates):
        numbers = dates[i].split('-')
        y = 0
        while y < len(numbers):
            if (int(numbers[y]) > 31) & (int(numbers[y]) < 1000) | (int(numbers[y]) == 0):
                x = int(numbers[y]) + 1900
                numbers[y] = str(x)
            y += 1
        dates[i]= date_format.join(numbers)
        
        if(re.match('^[0-9]{2}-[0-9]{2}-[0-9]{2}$', dates[i])):
            numbers[0] = str(int(numbers[0]) + 1900)
        dates[i]= date_format.join(numbers)
            
        if(int(numbers[0]) < int(numbers[2])):
            temp = numbers[0]
            numbers[0] = numbers[2]
            numbers[2] = temp
        dates[i]= date_format.join(numbers)
        i += 1

    data['date_of_birth'] = dates
    data['date_of_birth'] = data['date_of_birth'].replace(to_replace ='[\s]', value = '', regex = True)
    return data


# Funkcia sluzi na nahradenie NaN hodnot pre atribut age na základe roku narodenia
def fill_nan_age(data):
    new_values = []
    year = datetime.now().year
    i = 0
    while i < len(data.age):
        if (data.age.isnull()[i] == True):
            numbers = data.date_of_birth[i].split('-')
            new_values.append(year - int(numbers[0]))
        i += 1
    data.loc[data.age.isnull(), 'age'] = new_values
    return data


# Funkcia zmeni datovy typ atributu age z float na int
def convert_age_type(data):
    data['age'] = data.age.astype(int)
    return data


# Funkcia vymaze vsetky záznamy, ktorych datum narodenia je vačší ako aktuálny dátum
def drop_invalid_date(data):
    i = 0
    while i < len(data['date_of_birth']):
        datetime_object = datetime.strptime(data['date_of_birth'][i], '%Y-%m-%d')
        if(datetime_object > datetime.now()):
            data.drop(data.index[i], inplace=True)
        i += 1
    return data


# Funckia rozdelí jednu hodnotu atributu address na viacere hodnoty a vrati ich ako pd.Series
def split_address(text):
    address, tmp = text.replace('\r', '').split('\n')
    #APO - Army Post Office
    tmp = tmp.replace('APO', 'APO,')
    #FPO - Fleet Post Office
    tmp = tmp.replace('FPO', 'FPO,')
    #DPO - Diplomatic Post Office
    tmp = tmp.replace('DPO', 'DPO,')
    city, state = tmp.split(', ', 1)
    return pd.Series([address, city, state])


# Funkcia vykoná transformácie nad vstupným datasetom
def preprocessor(data):
    # Oprava formatu datumu narodenia
    data = fix_dates(data)

    # Doplnanie NaN hodnot v atribute age na zaklade datumu narodenia
    data = fill_nan_age(data)

    # Zmena datoveho typu atributu age z float na int
    data = convert_age_type(data)

    # Rozdelenie atributu address na atributy address/city/state
    data[['address', 'city', 'state']] = data.address.apply(split_address)

    # Fix formatu hodnôt - odstranenie bielych medzier
    data = fix_sex(data)

    # Odstranenie zaznamov, ktore nemaju validny datum narodenia
    data = drop_invalid_date(data)
    
    return data


# Funkcia nad vstupným trenovacim a validacnym datasetom zavola preprocesor a vráti ich vyčistené
def clean_personal(data_train, data_valid):
    data_train = preprocessor(data_train)
    data_valid = preprocessor(data_valid)
    return data_train, data_valid
