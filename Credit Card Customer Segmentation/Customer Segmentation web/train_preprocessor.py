import numpy as np 
import pandas as pd

def col_dropper(data, cols=['CUST_ID']):
    transformed_data = data.drop(columns=cols).copy()
    return transformed_data

def null_filler(data, col=['MINIMUM_PAYMENTS', 'CREDIT_LIMIT']):
    transformed_data = data.copy()
    for i in col:
        transformed_data[i].fillna(transformed_data[i].median(), inplace=True)
    return transformed_data

def log_transformer(data,cols = ['BALANCE', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 
        'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
        'CREDIT_LIMIT', 'PAYMENTS','MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT']):
    transformed_data = data.copy()
    for col in cols:
        transformed_data[col] = data[col].apply(lambda x: np.log(x) if x > 0  else 0)
    return transformed_data

def preprocessor(data):
    data = col_dropper(data)
    data1 = null_filler(data)
    data2 = log_transformer(data1)
    data3 = col_dropper(data2, cols=['CASH_ADVANCE_FREQUENCY', 'INSTALLMENTS_PURCHASES', 'PURCHASES_FREQUENCY'])
    return data3