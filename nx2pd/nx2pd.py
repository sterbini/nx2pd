from multiprocessing import Pool
import io
import time
from cern.nxcals.api.extraction.data.builders import *
import pyspark.sql.functions as func
from pyspark.sql.functions import col
from pyspark.sql.types import *
import numpy as np
import pandas as pd

import pytimber
ldb=pytimber.LoggingDB()
search=ldb.search
getUnit=ldb.getUnit
getDescription=ldb.getDescription

def _replace_specials(myString):
    if isinstance(myString,str):
        return myString.replace('.','@')
    if isinstance(myString,list):
        return [i.replace('.','@') for i in myString]
    else:
        assert False

def _invert_replace_specials(myString):
    if isinstance(myString,str):
        return myString.replace('@','.') # clearly the operation is invertible only under some hypothesis
    if isinstance(myString,list):
        return [i.replace('@','.') for i in myString]
    else:
        assert False
        
def _importNXCALS(variableName,t1, t2):
    ''' This hidden function takes a string a two pd datastamps. NXCALSsample is the fraction of rows to return.'''
    start_time = time.time()
    t1=t1.tz_convert('UTC').tz_localize(None)
    t2=t2.tz_convert('UTC').tz_localize(None)
    
    try:
        ds=DataQuery.builder(spark).byVariables().system('CMW').startTime(t1.strftime('%Y-%m-%d %H:%M:%S.%f')).endTime(t2.strftime('%Y-%m-%d %H:%M:%S.%f')).variable(variableName).buildDataset()
    except:
        try: 
            ds=DataQuery.builder(spark).byVariables().system('WINCCOA').startTime(t1.strftime('%Y-%m-%d %H:%M:%S.%f')).endTime(t2.strftime('%Y-%m-%d %H:%M:%S.%f')).variable(variableName).buildDataset()
        except:
            print('Variable not found in CMW and WINCCOA.')
    selectionStringDict={'int':{'value':'nxcals_value','label':'nxcals_value'}, 
                         'double':{'value':'nxcals_value','label':'nxcals_value'},
                         'float':{'value':'nxcals_value','label':'nxcals_value'},
                         'boolean':{'value':'nxcals_value','label':'nxcals_value'}, 
                         'string':{'value':'nxcals_value','label':'nxcals_value'},
                         'bigint':{'value':'nxcals_value','label':'nxcals_value'},
                         'struct<elements:array<int>,dimensions:array<int>>':{'value':'nxcals_value.elements','label':'elements'},
                         'struct<elements:array<double>,dimensions:array<int>>':{'value':'nxcals_value.elements','label':'elements'},
                         'struct<elements:array<float>,dimensions:array<int>>':{'value':'nxcals_value.elements','label':'elements'},}
    selectionStringValue=selectionStringDict[dict(ds.dtypes)['nxcals_value']]['value']
    selectionStringLabel=selectionStringDict[dict(ds.dtypes)['nxcals_value']]['label']
    aux=ds.select('nxcals_timestamp',selectionStringValue)
    return aux.withColumnRenamed('nxcals_timestamp','timestamp').withColumnRenamed(selectionStringLabel,_replace_specials(variableName))

def importNXCALS(inputList,t1,t2):
    outputList={}
    for i in inputList:
        outputList[i]={'pyspark df':_importNXCALS(i,t1,t2), 't1':t1, 't2':t2}
    out=pd.DataFrame(outputList).transpose()
    return out

def _join_df_list(pd_df,  on=["timestamp"], how='inner'):
    if isinstance(pd_df, pd.DataFrame):
        df=pd_df.iloc[0]['pyspark df']
        if len(pd_df)>1:
            for i in pd_df.iloc[1:].iterrows():
                df=df.join(i[1]['pyspark df'],on=on,how=how)
    if isinstance(pd_df, list):
        df=pd_df[0]
        if len(pd_df)>1:
            for i in pd_df[1:]:
                df=df.join(i,on=on,how=how)
    return df

def _parallelize(df, func, n_cores=1):
    '''To use multicores'''
    if n_cores==1:
        return func(df)
    else:
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
    return df

def timestamp_conversion_function(x):
    return x['timestamp'].apply(lambda x: pd.Timestamp(x).tz_localize('UTC'))

def _to_pandas(df, timestampConversion=False, sorted=False,n_cores=4):
    aux=df.toPandas()
    if 'timestamp' in aux.columns:
        aux=aux.set_index('timestamp'); aux.index.name=None
        if sorted:
            aux=aux.sort_index()
        if timestampConversion:
            aux['timestamp']=aux.index
            aux['timestamp']=_parallelize(aux,timestamp_conversion_function, n_cores=n_cores)
            aux.set_index('timestamp')
            aux=aux.set_index('timestamp'); aux.index.name=None
    myDict={}
    for i in aux.columns:
        myDict[i]=_invert_replace_specials(i)
    aux=aux.rename(columns=myDict)
    return aux
  
def _to_spark(df, timestampConversion=False, sorted=False,n_cores=4):
    my_df=df.copy(deep=True) #memory expensive
    my_df['timestamp']=my_df.index
    myDict={}
    for i in my_df.columns:
        myDict[i]=_replace_specials(i)
    my_df=my_df.rename(columns=myDict)
    aux=spark.createDataFrame(my_df)
    return aux
