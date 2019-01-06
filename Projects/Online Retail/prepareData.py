import pandas as pd
import warnings
import datetime as dt
from sklearn.preprocessing import StandardScaler

# Dataset:http://archive.ics.uci.edu/ml/datasets/Online+Retail

def prepareData(path):
  # data pre-processing
  warnings.filterwarnings('ignore')
  df = pd.read_excel(path)
  df1 = df
  df1.Country.nunique()
  df1.Country.unique()
  customer_country = df1[['Country', 'CustomerID']].drop_duplicates()
  customer_country.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID',
                                                                                                   ascending=False)
  df1.isnull().sum(axis=0)
  df1 = df1[pd.notnull(df1['CustomerID'])]
  df1 = df1[pd.notnull(df1['CustomerID'])]
  df1.Quantity.min()
  df1 = df1[(df1['Quantity'] > 0)]
  df1.shape

  def unique_counts(data):
    for i in data.columns:
      count = data[i].nunique()
  unique_counts(df1)
  df1['TotalPrice'] = df1['Quantity'] * df1['UnitPrice']
  df1['InvoiceDate'].min()
  df1['InvoiceDate'].max()
  NOW = dt.datetime(2011, 12, 10)
  df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'])
  rfmTable = df1.groupby('CustomerID').agg(
    {'InvoiceDate': lambda x: (NOW - x.max()).days, 'InvoiceNo': lambda x: len(x), 'TotalPrice': lambda x: x.sum()})
  rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
  rfmTable.rename(columns={'InvoiceDate': 'R', 'InvoiceNo': 'F', 'TotalPrice': 'M'}, inplace=True)
  return rfmTable


def standard(table):
  df_rfm_scaled = pd.DataFrame(StandardScaler().fit_transform(table)).columns = ['R', 'F', 'M']
  return df_rfm_scaled
