#!/usr/bin/env python
# coding: utf-8

#Import Module yang dibutuhkan
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pyodbc
import sklearn
import sklearn.metrics
import datetime
import math
import random
from datetime import date
from prophet import Prophet
from datetime import date
from dotenv.main import load_dotenv
import os

load_dotenv()
# Koneksi ke SQL Server
server = os.environ['SERVER']
database = os.environ['DATABASE']
username = os.environ['USERNAME_1']
password = os.environ['PASSWORD']
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()
query = "SELECT * FROM dwh.DM_production_transaction_Mine_Operation;"
df = pd.read_sql(query, cnxn)


#Preprocessing Data Coal Production
#Memilih data volume produksi batubara dengan ketentuan Product Type Name = "Mine Brand" dan "Market Brand"
df_temp_coal_real = df[df['Product Type Name'].isin(['Mine Brand','Market Brand'])].copy()
#hanya mengambil data real, tanpa RKAP maupun rakor. Jenis = Real merupakan gabungan dari hasil JS dan Ritase
df_temp_coal_real = df_temp_coal_real[df_temp_coal_real['Jenis']=='Real']
df_coal_real = df_temp_coal_real
#Menyusun df produksi batubara per area
df_coal_src_pivot = df_coal_real.pivot_table(
    index=['Tanggal','Loc Src Area Name'],
    values=['Tonase'],
    aggfunc={'Tonase': ['sum']}
)
df_coal_real_area_temp=df_coal_src_pivot.reset_index()
df_coal_real_area_temp.columns = ['{}'.format(col[0], col[1]) for col in df_coal_real_area_temp.columns]
#df produksi batubara daily per area
df_coal_real_area=df_coal_real_area_temp
#Preprocessing data dengan tidak menyertakan nilai kosong atau null
df_prep_coal = df_coal_real_area.copy().dropna(how='any')
# Merubah type Tanggal menjadi datetime
df_prep_coal['Tanggal'] = pd.to_datetime(df_prep_coal['Tanggal'])
#analisis hanya menggunakan data dimulai dari 1 januari 2020 (sesuai kesepakatan dengan PTBA)
df_prep_coal = df_prep_coal[df_prep_coal['Tanggal'] >= '2020-01-01']
# menyusun kolom berdasarkan Loc Src Area Name dan mengurutkan berdasarka tanggal paling awal
temp_coal_area= df_prep_coal.groupby(['Tanggal','Loc Src Area Name'])['Tonase'].sum().reset_index().sort_values(by=['Tanggal'])
# Tidak menyertakan data dengan nilai 0
temp_coal_area_clean = temp_coal_area [temp_coal_area ['Tonase'] != 0].reset_index(drop=True).sort_values(by=['Tanggal']).reset_index(drop=True)
# df produksi batubara yang sudah selesai preprocessing (siap digunakan untuk forecasting)
coal_area_clean = temp_coal_area_clean[temp_coal_area_clean['Tanggal'] <= pd.to_datetime('today')]


#Untuk mengatasi adanya outlier, digunakan metode normalisasi data
#normalization function
def norm(x,data):
    xn = (x - min(data)) / (1.5*max(data) - min(data))
    return xn

#invers normalization function. digunakan untuk merubah data hasil normalisasi ke data asli
def inv_norm(xn,data) :
    x = (xn*(1.5*max(data) - min(data))) + min(data)
    return x

# Melakukan Stack Normalization Data 
def stack_norm(df,target):
    timestamp = df['ds']
    X = df.drop(['ds',target],axis=1)
    
    res = []
    for i in range(len(X.columns)):
        temp =  np.array(norm(X[X.columns[i]],X[X.columns[i]]))
        res.append(temp)

    # horizontally stack columns
    dataset = pd.DataFrame(res).T
    dataset.columns = X.columns
    
    y = df[target]
    y = pd.DataFrame({'y':np.array(norm(y,y))}) # <-----------TARGET
    
    
    df_out = pd.concat([timestamp,dataset,y],axis=1)
    df_out = df_out.rename(columns = {'Tanggal':'ds'})
    
    return df_out

#merubah nama Tanggal menjadi ds dan target/nilai yang akan diforecast menjadi y
#hal ini bertujuan untuk menyesuaikan dengan fungsi metode forecast yang digunakan
def prophet_rename(df,date_column,target):
    df = df.rename(columns = {date_column:'ds', target:'y'})
    return df

#Menentukan kombinasi parameter metode forecasting yang akan digunakan
from sklearn.model_selection import ParameterGrid
params_grid = {'growth': ['linear', 'logistic', 'flat'],
               'changepoint_prior_scale':[0.1],
              'n_changepoints' : [100,150]}

grid = ParameterGrid(params_grid)
cnt = 0
for p in grid:
    cnt = cnt+1

print('Total Possible Models',cnt)

#membuat dictionary nama Loc Src Area Name yang digunakan
loc_coal_df_list = {}
for i in coal_area_clean['Loc Src Area Name'].unique():
    temp_df = coal_area_clean[coal_area_clean['Loc Src Area Name'] == i].reset_index(drop=True).drop(['Loc Src Area Name'],axis=1)
    loc_coal_df_list[i] = temp_df

###########################################################################################################################    
#tahapan Forecasting Menggunakan FB Prophet    
#panjang data testing yang digunakan. 0,1 menandakan 10% dari data digunakan sebagai testing
test_size = 0.1

prophet_df_dict = {}
evaluation_df = []
result_parameter = []
loc_hyperparameter = {}

for i in list(loc_coal_df_list.keys()):
    temp_name = prophet_rename(loc_coal_df_list[i],'Tanggal','Tonase')
    temp_dict = stack_norm(temp_name,'y')
    temp_date = temp_dict
    
    # Train and Test

    train = temp_date.iloc[:int((1-test_size)* len(temp_date))]
    test = temp_date.iloc[int((test_size*len(temp_date))):]
    
    model_parameters = pd.DataFrame(columns = ['SMAPE','MAPE','Parameters'])
    
    for params in grid:
        print(params)
        random.seed(0)

        # Build model

        model_forecast = Prophet(changepoint_prior_scale = params['changepoint_prior_scale'],
                         n_changepoints = params['n_changepoints'],
                         seasonality_mode = 'additive',
                         weekly_seasonality = True,
                         daily_seasonality = True,
                         yearly_seasonality = True,
                         interval_width = 0.95)
        model_forecast.add_country_holidays(country_name='Indonesia')
        model_forecast.fit(train)

        forecast = model_forecast.predict(test)
        forecast_result = forecast[['ds','yhat']]
        merged = test.merge(forecast_result, on=['ds'])

        # Inverse merged

        yhat = pd.DataFrame({'yhat':inv_norm(merged['yhat'], temp_name['y'])})
        yval = pd.DataFrame({'y':inv_norm(merged['y'], temp_name['y'])})

        merged['yhat'] = yhat
        merged['y'] = yval

        
        def sMAPE_formula(A, F):
            numerator = np.abs(A - F)
            denominator = (np.abs(A) + np.abs(F)) / 2
            return np.mean(numerator / denominator) * 100


        # report performance
        
        smape = sMAPE_formula(merged['y'], merged['yhat'])
        #mape = MAPE_formula(merged['y'], merged['yhat'])
        
        print('Symetrical Mean Absolute Percentage Error(SMAPE) = ',smape,str(' %'))
        print('Loc_src_area_name: ',i)
        model_parameters = model_parameters.append({'SMAPE':smape,'Parameters':params},ignore_index=True)
        
        prophet_df_dict[i] = merged
    
        # Hyperparameter result dataframe

    loc_hyperparameter[i] = model_parameters.sort_values(by=['SMAPE'])
    


#df hasil evaluasi modelling
df_eval_coal = []
for i in list(loc_hyperparameter.keys()):
 df_eval_coal.append(loc_hyperparameter[i][:1][['SMAPE']])
 temp_eval_coal = pd.concat(df_eval_coal).reset_index(drop=True)
forecast_coal_df = pd.DataFrame({'Loc Src Area Name':list(loc_hyperparameter.keys())})
eval_df_coal_final = pd.concat([forecast_coal_df,temp_eval_coal],axis=1)
#margin error dari model yang dibentuk (per area)
ME_coal_temp=eval_df_coal_final.assign(eval='ME_Coal')
#margin error untuk all area dari hasil forecast
smape_coal_all_temp = sMAPE_formula(merged['y'], merged['yhat'])
smape_coal_all_temp = {'Loc Src Area Name': ['All_Area'], 'SMAPE': [smape_coal_all_temp],'eval':['ME_Coal']}
smape_coal_all=pd.DataFrame(smape_coal_all_temp)
#margin error area dan all (nilai dalam persen)
ME_Coal = pd.concat([smape_coal_all,ME_coal_temp], axis=0, ignore_index=True)
#List nilai parameter yang digunakan, berdasarkan nilai sMAPE minimal
list_params_coal = []
for j in list(loc_hyperparameter.keys()):
 temp = list(loc_hyperparameter[j][:1]['Parameters'].values)
 for i in temp:
     list_params_coal.append(i)
temp_df_param_coal = pd.DataFrame(list_params_coal)
df_params_coal_temp = pd.concat([forecast_coal_df,temp_df_param_coal],axis=1)
df_params_coal=df_params_coal_temp.assign(eval='Param_coal')


#fungsi filter ini hanya membatasi 4 tahun data terakhir yang akan ditampilkan.
#dan jumlah forecasting dibatasi pada 300 hari kedepan
days = 300
def filter_date(df,total_years=4):
    # Max Date
    max_date = df['ds'].max()
    
    # Limit Two years
    years_ago = max_date - pd.DateOffset(years=total_years)
    
    # Filter
    df_filtered = df[(df['ds'] >= years_ago) & (df['ds'] <= max_date)]
    
    return df_filtered

#Tahapan deploy untuk mengaplikasikan model forecasting berdasarkan parameter terbaik yang telah dibentuk di tahahapan sebelumnya

prophet_df_coal_dict_deploy = {}
for i in list(loc_coal_df_list.keys()):
    temp_name = prophet_rename(loc_coal_df_list[i],'Tanggal','Tonase')
    temp_name = temp_name[['ds','y']]
    temp_dict = stack_norm(temp_name,'y')
    temp_date = filter_date(temp_dict, total_years = 4)

    # Build model
    model_forecast = Prophet(changepoint_prior_scale = df_params_coal[df_params_coal['Loc Src Area Name']==i]['changepoint_prior_scale'].values[0],
                     n_changepoints = df_params_coal[df_params_coal['Loc Src Area Name']==i]['n_changepoints'].values[0],
                     weekly_seasonality = True,
                     daily_seasonality = True,
                     yearly_seasonality = True,
                     interval_width = 0.95)
    model_forecast.add_country_holidays(country_name='Indonesia')
    model_forecast.fit(temp_date)
    
    # Predict
    future = model_forecast.make_future_dataframe(periods=days)
    fcst = model_forecast.predict(future)
    forecast_coal_result = fcst[['ds','yhat']]
    
    # Inverse Normalization
    yhat = pd.DataFrame({'yhat':inv_norm(forecast_coal_result['yhat'], temp_name['y'])})
    forecast_coal_result['yhat'] = yhat
    prophet_df_coal_dict_deploy[i] = forecast_coal_result


# Export Dataset
datamart_actual_coal = []
for loc_src_area_name in (loc_coal_df_list.keys()):
    temp_dm = loc_coal_df_list[loc_src_area_name]
    temp_dm['Loc Src Area Name'] = loc_src_area_name
    datamart_actual_coal.append(temp_dm)
dm_actual_coal = pd.concat(datamart_actual_coal).reset_index(drop=True)


#DM yang berisi nilai hasil forecast coal
datamart_pred_coal = []
for loc_src_area_name in (loc_coal_df_list.keys()):
    temp_dm = prophet_df_coal_dict_deploy[loc_src_area_name]
    temp_dm['Loc Src Area Name'] = loc_src_area_name
    temp_dm['Tonase'] = np.round(temp_dm['yhat'])
    datamart_pred_coal.append(temp_dm)
dm_pred_coal_temp = pd.concat(datamart_pred_coal).reset_index(drop=True)
dm_pred_coal= dm_pred_coal_temp.assign(Tipe='forecast', Jenis='forecast', Product_Type_Name='coal forecast')



#Preprocessing Data Production dari masing-masing contractor
#Memilih data volume produksi batubara dengan ketentuan Product Type Name = "Mine Brand" dan "Market Brand"
#Memilih data sesuai nama kontraktor
df_temp_contractor_real = df[df['Product Type Name'].isin(['Mine Brand','Market Brand'])].copy()
df_temp_contractor_real = df_temp_contractor_real[df_temp_contractor_real['Contractor Name'].isin(['PT PAMA PERSADA NUSANTARA', 'PT SATRIA BAHANA SARANA','SWAKELOLA'])]

#hanya mengambil data real, tanpa RKAP maupun rakor. Jenis = Real merupakan gabungan dari hasil JS dan Ritase
df_temp_contractor_real = df_temp_contractor_real[df_temp_contractor_real['Jenis']=='Real']
df_contractor_real = df_temp_contractor_real

#Menyusun df produksi batubara per area
df_contractor_src_pivot = df_contractor_real.pivot_table(
    index=['Tanggal','Contractor Name'],
    values=['Tonase'],
    aggfunc={'Tonase': ['sum']}
)
df_contractor_real_area_temp=df_contractor_src_pivot.reset_index()
df_contractor_real_area_temp.columns = ['{}'.format(col[0], col[1]) for col in df_contractor_real_area_temp.columns]
#df produksi batubara daily per area
df_contractor_real_area=df_contractor_real_area_temp

###################################################################
#Preprocessing data dengan tidak menyertakan nilai kosong atau null
df_prep_contractor = df_contractor_real_area.copy().dropna(how='any')
# Merubah type Tanggal menjadi datetime
df_prep_contractor['Tanggal'] = pd.to_datetime(df_prep_contractor['Tanggal'])
#analisis hanya menggunakan data dimulai dari 1 januari 2020 (sesuai kesepakatan dengan PTBA)
df_prep_contractor = df_prep_contractor[df_prep_contractor['Tanggal'] >= '2020-01-01']
# menyusun kolom berdasarkan Loc Src Area Name dan mengurutkan berdasarka tanggal paling awal
temp_contractor_area= df_prep_contractor.groupby(['Tanggal','Contractor Name'])['Tonase'].sum().reset_index().sort_values(by=['Tanggal'])
# Tidak menyertakan data dengan nilai 0
temp_contractor_area_clean = temp_contractor_area [temp_contractor_area ['Tonase'] != 0].reset_index(drop=True).sort_values(by=['Tanggal']).reset_index(drop=True)
# df produksi batubara yang sudah selesai preprocessing (siap digunakan untuk forecasting)
contractor_area_clean = temp_contractor_area_clean[temp_contractor_area_clean['Tanggal'] <= pd.to_datetime('today')]



#membuat dictionary nama Loc Src Area Name yang digunakan
loc_contractor_df_list = {}
for i in contractor_area_clean['Contractor Name'].unique():
    temp_df = contractor_area_clean[contractor_area_clean['Contractor Name'] == i].reset_index(drop=True).drop(['Contractor Name'],axis=1)
    loc_contractor_df_list[i] = temp_df

###########################################################################################################################    
#tahapan Forecasting Menggunakan FB Prophet    
#panjang data testing yang digunakan. 0,1 menandakan 10% dari data digunakan sebagai testing
test_size = 0.1

prophet_df_dict = {}
evaluation_df = []
result_parameter = []
loc_hyperparameter = {}

for i in list(loc_contractor_df_list.keys()):
    temp_name = prophet_rename(loc_contractor_df_list[i],'Tanggal','Tonase')
    temp_dict = stack_norm(temp_name,'y')
    temp_date = temp_dict
    
    # Train and Test

    train = temp_date.iloc[:int((1-test_size)* len(temp_date))]
    test = temp_date.iloc[int((test_size*len(temp_date))):]
    
    model_parameters = pd.DataFrame(columns = ['SMAPE','MAPE','Parameters'])
    
    for params in grid:
        print(params)
        random.seed(0)

        # Build model

        model_forecast = Prophet(changepoint_prior_scale = params['changepoint_prior_scale'],
                         n_changepoints = params['n_changepoints'],
                         seasonality_mode = 'additive',
                         weekly_seasonality = True,
                         daily_seasonality = True,
                         yearly_seasonality = True,
                         interval_width = 0.95)
        model_forecast.add_country_holidays(country_name='Indonesia')
        model_forecast.fit(train)

        forecast = model_forecast.predict(test)
        forecast_result = forecast[['ds','yhat']]
        merged = test.merge(forecast_result, on=['ds'])

        # Inverse merged

        yhat = pd.DataFrame({'yhat':inv_norm(merged['yhat'], temp_name['y'])})
        yval = pd.DataFrame({'y':inv_norm(merged['y'], temp_name['y'])})

        merged['yhat'] = yhat
        merged['y'] = yval

        
        def sMAPE_formula(A, F):
            numerator = np.abs(A - F)
            denominator = (np.abs(A) + np.abs(F)) / 2
            return np.mean(numerator / denominator) * 100
        
        
        def MAPE_formula(A, F):
            return np.mean(np.abs((A - F) / A)) * 100


        # report performance
        
        smape = sMAPE_formula(merged['y'], merged['yhat'])
        #mape = MAPE_formula(merged['y'], merged['yhat'])
        
        print('Symetrical Mean Absolute Percentage Error(SMAPE) = ',smape,str(' %'))
        #print('Mean Absolute Percentage Error(MAPE) = ',mape,str(' %'))
        print('Contractor Name: ',i)
        model_parameters = model_parameters.append({'SMAPE':smape,'Parameters':params},ignore_index=True)
        
        prophet_df_dict[i] = merged
    
        # Hyperparameter result dataframe

    loc_hyperparameter[i] = model_parameters.sort_values(by=['SMAPE'])


#df hasil evaluasi modelling
df_eval_contractor = []
for i in list(loc_hyperparameter.keys()):
    df_eval_contractor.append(loc_hyperparameter[i][:1][['SMAPE']])
    temp_eval_contractor = pd.concat(df_eval_contractor).reset_index(drop=True)
forecast_contractor_df = pd.DataFrame({'Contractor Name':list(loc_hyperparameter.keys())})
eval_df_contractor_final = pd.concat([forecast_contractor_df,temp_eval_contractor],axis=1)
#margin error dari model yang dibentuk (per area)
ME_contractor_temp=eval_df_contractor_final.assign(eval='ME_contractor')

#margin error untuk all area dari hasil forecast
smape_contractor_all_temp = sMAPE_formula(merged['y'], merged['yhat'])
smape_contractor_all_temp = {'Contractor Name': ['All_Area'], 'SMAPE': [smape_contractor_all_temp],'eval':['ME_contractor']}
smape_contractor_all=pd.DataFrame(smape_contractor_all_temp)
#margin error area dan all (nilai dalam persen)
ME_contractor = pd.concat([smape_contractor_all,ME_contractor_temp], axis=0, ignore_index=True)
#List nilai parameter yang digunakan, berdasarkan nilai sMAPE minimal
list_params_contractor = []
for j in list(loc_hyperparameter.keys()):
    temp = list(loc_hyperparameter[j][:1]['Parameters'].values)
    for i in temp:
        list_params_contractor.append(i)
temp_df_param_contractor = pd.DataFrame(list_params_contractor)

df_params_contractor_temp = pd.concat([forecast_contractor_df,temp_df_param_contractor],axis=1)
df_params_contractor=df_params_contractor_temp.assign(eval='Param_contractor')


#fungsi filter ini hanya membatasi 4 tahun data terakhir yang akan ditampilkan.
#dan jumlah forecasting dibatasi pada 300 hari kedepan
days = 300
def filter_date(df,total_years=4):
    # Max Date
    max_date = df['ds'].max()
    
    # Limit Two years
    years_ago = max_date - pd.DateOffset(years=total_years)
    
    # Filter
    df_filtered = df[(df['ds'] >= years_ago) & (df['ds'] <= max_date)]
    
    return df_filtered

#Tahapan deploy untuk mengaplikasikan model forecasting berdasarkan parameter terbaik yang telah dibentuk di tahahapan sebelumnya

prophet_df_contractor_dict_deploy = {}
for i in list(loc_contractor_df_list.keys()):
    temp_name = prophet_rename(loc_contractor_df_list[i],'Tanggal','Tonase')
    temp_name = temp_name[['ds','y']]
    temp_dict = stack_norm(temp_name,'y')
    temp_date = filter_date(temp_dict, total_years = 4)

    # Build model
    model_forecast = Prophet(changepoint_prior_scale = df_params_contractor[df_params_contractor['Contractor Name']==i]['changepoint_prior_scale'].values[0],
                     n_changepoints = df_params_contractor[df_params_contractor['Contractor Name']==i]['n_changepoints'].values[0],
                     weekly_seasonality = True,
                     daily_seasonality = True,
                     yearly_seasonality = True,
                     interval_width = 0.95)
    model_forecast.add_country_holidays(country_name='Indonesia')
    model_forecast.fit(temp_date)
    
    # Predict
    future = model_forecast.make_future_dataframe(periods=days)
    fcst = model_forecast.predict(future)
    forecast_contractor_result = fcst[['ds','yhat']]
    
    # Inverse Normalization
    yhat = pd.DataFrame({'yhat':inv_norm(forecast_contractor_result['yhat'], temp_name['y'])})
    forecast_contractor_result['yhat'] = yhat
    prophet_df_contractor_dict_deploy[i] = forecast_contractor_result


# Export Dataset
datamart_actual_contractor = []
for contractor_name in (loc_contractor_df_list.keys()):
    temp_dm = loc_contractor_df_list[contractor_name]
    temp_dm['Contractor Name'] = contractor_name
    datamart_actual_contractor.append(temp_dm)
dm_actual_contractor = pd.concat(datamart_actual_contractor).reset_index(drop=True)


#DM yang berisi nilai hasil forecast contractor
datamart_pred_contractor = []
for contractor_name in (loc_contractor_df_list.keys()):
    temp_dm = prophet_df_contractor_dict_deploy[contractor_name]
    temp_dm['Contractor Name'] = contractor_name
    temp_dm['Tonase'] = np.round(temp_dm['yhat'])
    datamart_pred_contractor.append(temp_dm)
dm_pred_contractor_temp = pd.concat(datamart_pred_contractor).reset_index(drop=True)
dm_pred_contractor= dm_pred_contractor_temp.assign(Tipe='forecast', Jenis='forecast', Product_Type_Name='contractor forecast')
dm_pred_contractor

#Preprocessing Data Overburden
#Memilih data volume OBdengan ketentuan Product Type Name = "Waste"
df_temp_OB_real = df[df['Product Type Name'].isin(['Waste'])].copy()

#hanya mengambil data real, tanpa RKAP maupun rakor. Jenis = Real 
df_temp_OB_real = df_temp_OB_real[df_temp_OB_real['Jenis']=='Real']
df_OB_real = df_temp_OB_real

#Menyusun df OB per area
df_OB_src_pivot = df_OB_real.pivot_table(
    index=['Tanggal','Loc Src Area Name'],
    values=['Tonase'],
    aggfunc={'Tonase': ['sum']}
)
df_OB_real_area_temp=df_OB_src_pivot.reset_index()
df_OB_real_area_temp.columns = ['{}'.format(col[0], col[1]) for col in df_OB_real_area_temp.columns]
#df produksi batubara daily per area
df_OB_real_area=df_OB_real_area_temp

###################################################################
#Preprocessing data dengan tidak menyertakan nilai kosong atau null
df_prep_OB = df_OB_real_area.copy().dropna(how='any')
# Merubah type Tanggal menjadi datetime
df_prep_OB['Tanggal'] = pd.to_datetime(df_prep_OB['Tanggal'])
#analisis hanya menggunakan data dimulai dari 1 januari 2020 (sesuai kesepakatan dengan PTBA)
df_prep_OB = df_prep_OB[df_prep_OB['Tanggal'] >= '2020-01-01']
# menyusun kolom berdasarkan Loc Src Area Name dan mengurutkan berdasarka tanggal paling awal
temp_OB_area= df_prep_OB.groupby(['Tanggal','Loc Src Area Name'])['Tonase'].sum().reset_index().sort_values(by=['Tanggal'])
# Tidak menyertakan data dengan nilai 0
temp_OB_area_clean = temp_OB_area [temp_OB_area ['Tonase'] != 0].reset_index(drop=True).sort_values(by=['Tanggal']).reset_index(drop=True)
# df produksi batubara yang sudah selesai preprocessing (siap digunakan untuk forecasting)
OB_area_clean = temp_OB_area_clean[temp_OB_area_clean['Tanggal'] <= pd.to_datetime('today')]


#membuat dictionary nama Loc Src Area Name yang digunakan
loc_OB_df_list = {}
for i in OB_area_clean['Loc Src Area Name'].unique():
    temp_df = OB_area_clean[OB_area_clean['Loc Src Area Name'] == i].reset_index(drop=True).drop(['Loc Src Area Name'],axis=1)
    loc_OB_df_list[i] = temp_df

###########################################################################################################################    
#tahapan Forecasting Menggunakan FB Prophet    
#panjang data testing yang digunakan. 0,1 menandakan 10% dari data digunakan sebagai testing
test_size = 0.1

prophet_df_dict = {}
evaluation_df = []
result_parameter = []
loc_hyperparameter = {}

for i in list(loc_OB_df_list.keys()):
    temp_name = prophet_rename(loc_OB_df_list[i],'Tanggal','Tonase')
    temp_dict = stack_norm(temp_name,'y')
    temp_date = temp_dict
    
    # Train and Test

    train = temp_date.iloc[:int((1-test_size)* len(temp_date))]
    test = temp_date.iloc[int((test_size*len(temp_date))):]
    
    model_parameters = pd.DataFrame(columns = ['SMAPE','MAPE','Parameters'])
    
    for params in grid:
        print(params)
        random.seed(0)

        # Build model

        model_forecast = Prophet(changepoint_prior_scale = params['changepoint_prior_scale'],
                         n_changepoints = params['n_changepoints'],
                         seasonality_mode = 'additive',
                         weekly_seasonality = True,
                         daily_seasonality = True,
                         yearly_seasonality = True,
                         interval_width = 0.95)
        model_forecast.add_country_holidays(country_name='Indonesia')
        model_forecast.fit(train)

        forecast = model_forecast.predict(test)
        forecast_result = forecast[['ds','yhat']]
        merged = test.merge(forecast_result, on=['ds'])

        # Inverse merged

        yhat = pd.DataFrame({'yhat':inv_norm(merged['yhat'], temp_name['y'])})
        yval = pd.DataFrame({'y':inv_norm(merged['y'], temp_name['y'])})

        merged['yhat'] = yhat
        merged['y'] = yval

        
        def sMAPE_formula(A, F):
            numerator = np.abs(A - F)
            denominator = (np.abs(A) + np.abs(F)) / 2
            return np.mean(numerator / denominator) * 100
        
        
        def MAPE_formula(A, F):
            return np.mean(np.abs((A - F) / A)) * 100


        # report performance
        
        smape = sMAPE_formula(merged['y'], merged['yhat'])
        #mape = MAPE_formula(merged['y'], merged['yhat'])
        
        print('Symetrical Mean Absolute Percentage Error(SMAPE) = ',smape,str(' %'))
        #print('Mean Absolute Percentage Error(MAPE) = ',mape,str(' %'))
        print('Loc_src_area_name: ',i)
        model_parameters = model_parameters.append({'SMAPE':smape,'Parameters':params},ignore_index=True)
        
        prophet_df_dict[i] = merged
    
        # Hyperparameter result dataframe

    loc_hyperparameter[i] = model_parameters.sort_values(by=['SMAPE'])


#df hasil evaluasi modelling
df_eval_OB = []
for i in list(loc_hyperparameter.keys()):
    df_eval_OB.append(loc_hyperparameter[i][:1][['SMAPE']])
    temp_eval_OB = pd.concat(df_eval_OB).reset_index(drop=True)
forecast_OB_df = pd.DataFrame({'Loc Src Area Name':list(loc_hyperparameter.keys())})
eval_df_OB_final = pd.concat([forecast_OB_df,temp_eval_OB],axis=1)
#margin error dari model yang dibentuk (per area)
ME_OB_temp=eval_df_OB_final.assign(eval='ME_OB')

#margin error untuk all area dari hasil forecast
smape_OB_all_temp = sMAPE_formula(merged['y'], merged['yhat'])
smape_OB_all_temp = {'Loc Src Area Name': ['All_Area'], 'SMAPE': [smape_OB_all_temp],'eval':['ME_OB']}
smape_OB_all=pd.DataFrame(smape_OB_all_temp)
#margin error area dan all (nilai dalam persen)
ME_OB = pd.concat([smape_OB_all,ME_OB_temp], axis=0, ignore_index=True)
#List nilai parameter yang digunakan, berdasarkan nilai sMAPE minimal
list_params_OB = []
for j in list(loc_hyperparameter.keys()):
    temp = list(loc_hyperparameter[j][:1]['Parameters'].values)
    for i in temp:
        list_params_OB.append(i)
temp_df_param_OB = pd.DataFrame(list_params_OB)
df_params_OB_temp = pd.concat([forecast_OB_df,temp_df_param_OB],axis=1)
df_params_OB=df_params_OB_temp.assign(eval='Param_OB')



#Tahapan deploy untuk mengaplikasikan model forecasting berdasarkan parameter terbaik yang telah dibentuk di tahahapan sebelumnya
prophet_df_OB_dict_deploy = {}
for i in list(loc_OB_df_list.keys()):
    temp_name = prophet_rename(loc_OB_df_list[i],'Tanggal','Tonase')
    temp_name = temp_name[['ds','y']]
    temp_dict = stack_norm(temp_name,'y')
    temp_date = filter_date(temp_dict, total_years = 4)

    # Build model
    model_forecast = Prophet(changepoint_prior_scale = df_params_OB[df_params_OB['Loc Src Area Name']==i]['changepoint_prior_scale'].values[0],
                     n_changepoints = df_params_OB[df_params_OB['Loc Src Area Name']==i]['n_changepoints'].values[0],
                     weekly_seasonality = True,
                     daily_seasonality = True,
                     yearly_seasonality = True,
                     interval_width = 0.95)
    model_forecast.add_country_holidays(country_name='Indonesia')
    model_forecast.fit(temp_date)
    # Predict
    future = model_forecast.make_future_dataframe(periods=days)
    fcst = model_forecast.predict(future)
    forecast_OB_result = fcst[['ds','yhat']]
    # Inverse Normalization
    yhat = pd.DataFrame({'yhat':inv_norm(forecast_OB_result['yhat'], temp_name['y'])})
    forecast_OB_result['yhat'] = yhat
    prophet_df_OB_dict_deploy[i] = forecast_OB_result

# Export Dataset
datamart_actual_OB = []
for loc_src_area_name in (loc_OB_df_list.keys()):
    temp_dm = loc_OB_df_list[loc_src_area_name]
    temp_dm['Loc Src Area Name'] = loc_src_area_name
    datamart_actual_OB.append(temp_dm)
dm_actual_OB = pd.concat(datamart_actual_OB).reset_index(drop=True)
#DM yang berisi nilai hasil forecast OB
datamart_pred_OB = []
for loc_src_area_name in (loc_OB_df_list.keys()):
    temp_dm = prophet_df_OB_dict_deploy[loc_src_area_name]
    temp_dm['Loc Src Area Name'] = loc_src_area_name
    temp_dm['Tonase'] = np.round(temp_dm['yhat'])
    datamart_pred_OB.append(temp_dm)
dm_pred_OB_temp = pd.concat(datamart_pred_OB).reset_index(drop=True)
dm_pred_OB= dm_pred_OB_temp.assign(Tipe='forecast', Jenis='forecast', Product_Type_Name='OB forecast')


##STRIPPING RATIO
#Preprocessing Data Stripping Ratio
coal_cleans_temp=coal_area_clean.assign(Type='Coal')
OB_cleans_temp=OB_area_clean.assign(Type='OB')
df_temp_SR_real = pd.concat([coal_cleans_temp,OB_cleans_temp], axis=0, ignore_index=True)

# Menghitung nilai tonase berdasarkan pembagian antara Type A dan Type B
df_temp_SR = df_temp_SR_real.groupby(['Tanggal', 'Loc Src Area Name']).apply(lambda x: x['Tonase'][x['Type'] == 'OB'].sum() / x['Tonase'][x['Type'] == 'Coal'].sum())
df_SR=df_temp_SR.reset_index()
df_SR.rename(columns={0: 'Tonase'}, inplace=True)
#df SR daily per area
df_SR_real_area=df_SR
###################################################################
#Preprocessing data dengan tidak menyertakan nilai kosong atau null
df_prep_SR = df_SR_real_area.copy().dropna(how='any')

# Merubah type Tanggal menjadi datetime
df_prep_SR['Tanggal'] = pd.to_datetime(df_prep_SR['Tanggal'])

#analisis hanya menggunakan data dimulai dari 1 januari 2020 (sesuai kesepakatan dengan PTBA)
df_prep_SR = df_prep_SR[df_prep_SR['Tanggal'] >= '2020-01-01']

# menyusun kolom berdasarkan Loc Src Area Name dan mengurutkan berdasarka tanggal paling awal
temp_SR_area= df_prep_SR.groupby(['Tanggal','Loc Src Area Name'])['Tonase'].sum().reset_index().sort_values(by=['Tanggal'])
# Tidak menyertakan data dengan nilai 0
temp_SR_area_clean = temp_SR_area [temp_SR_area ['Tonase'] != 0].reset_index(drop=True).sort_values(by=['Tanggal']).reset_index(drop=True)
# df produksi batubara yang sudah selesai preprocessing (siap digunakan untuk forecasting)
SR_area_clean = temp_SR_area_clean[temp_SR_area_clean['Tanggal'] <= pd.to_datetime('today')]


#membuat dictionary nama Loc Src Area Name yang digunakan dari data SR clean
loc_SR_df_list = {}
for i in SR_area_clean['Loc Src Area Name'].unique():
    temp_df = SR_area_clean[SR_area_clean['Loc Src Area Name'] == i].reset_index(drop=True).drop(['Loc Src Area Name'],axis=1)
    loc_SR_df_list[i] = temp_df
###########################################################################################################################    
#tahapan Forecasting Menggunakan FB Prophet    
#panjang data testing yang digunakan. 0,1 menandakan 10% dari data digunakan sebagai testing
test_size = 0.1

prophet_df_dict = {}
evaluation_df = []
result_parameter = []
loc_hyperparameter = {}

for i in list(loc_SR_df_list.keys()):
    temp_name = prophet_rename(loc_SR_df_list[i],'Tanggal','Tonase')
    temp_dict = stack_norm(temp_name,'y')
    temp_date = temp_dict
    
    # Train and Test

    train = temp_date.iloc[:int((1-test_size)* len(temp_date))]
    test = temp_date.iloc[int((test_size*len(temp_date))):]
    
    model_parameters = pd.DataFrame(columns = ['SMAPE','MAPE','Parameters'])
    
    for params in grid:
        print(params)
        random.seed(0)

        # Build model

        model_forecast = Prophet(changepoint_prior_scale = params['changepoint_prior_scale'],
                         n_changepoints = params['n_changepoints'],
                         seasonality_mode = 'additive',
                         weekly_seasonality = True,
                         daily_seasonality = True,
                         yearly_seasonality = True,
                         interval_width = 0.95)
        model_forecast.add_country_holidays(country_name='Indonesia')
        model_forecast.fit(train)

        forecast = model_forecast.predict(test)
        forecast_result = forecast[['ds','yhat']]
        merged = test.merge(forecast_result, on=['ds'])

        # Inverse merged

        yhat = pd.DataFrame({'yhat':inv_norm(merged['yhat'], temp_name['y'])})
        yval = pd.DataFrame({'y':inv_norm(merged['y'], temp_name['y'])})

        merged['yhat'] = yhat
        merged['y'] = yval

        
        def sMAPE_formula(A, F):
            numerator = np.abs(A - F)
            denominator = (np.abs(A) + np.abs(F)) / 2
            return np.mean(numerator / denominator) * 100
        
        def MAPE_formula(A, F):
            return np.mean(np.abs((A - F) / A)) * 100


        # report performance
        
        smape = sMAPE_formula(merged['y'], merged['yhat'])
        #mape = MAPE_formula(merged['y'], merged['yhat'])
        
        print('Symetrical Mean Absolute Percentage Error(SMAPE) = ',smape,str(' %'))
        #print('Mean Absolute Percentage Error(MAPE) = ',mape,str(' %'))
        print('Loc_src_area_name: ',i)
        model_parameters = model_parameters.append({'SMAPE':smape,'Parameters':params},ignore_index=True)
        
        prophet_df_dict[i] = merged
    
        # Hyperparameter result dataframe

    loc_hyperparameter[i] = model_parameters.sort_values(by=['SMAPE'])
    

#df hasil evaluasi modelling
df_eval_SR = []
for i in list(loc_hyperparameter.keys()):
    df_eval_SR.append(loc_hyperparameter[i][:1][['SMAPE']])
    temp_eval_SR = pd.concat(df_eval_SR).reset_index(drop=True)
forecast_SR_df = pd.DataFrame({'Loc Src Area Name':list(loc_hyperparameter.keys())})
eval_df_SR_final = pd.concat([forecast_SR_df,temp_eval_SR],axis=1)
#margin error dari model yang dibentuk (per area)
ME_SR_temp=eval_df_SR_final.assign(eval='ME_SR')

#margin error untuk all area dari hasil forecast
smape_SR_all_temp = sMAPE_formula(merged['y'], merged['yhat'])
smape_SR_all_temp = {'Loc Src Area Name': ['All_Area'], 'SMAPE': [smape_SR_all_temp],'eval':['ME_SR']}
smape_SR_all=pd.DataFrame(smape_SR_all_temp)
#margin error area dan all (nilai dalam persen)
ME_SR = pd.concat([smape_SR_all,ME_SR_temp], axis=0, ignore_index=True)
#List nilai parameter yang digunakan, berdasarkan nilai sMAPE minimal
list_params_SR = []
for j in list(loc_hyperparameter.keys()):
    temp = list(loc_hyperparameter[j][:1]['Parameters'].values)
    for i in temp:
        list_params_SR.append(i)
temp_df_param_SR = pd.DataFrame(list_params_SR)
df_params_SR_temp = pd.concat([forecast_SR_df,temp_df_param_SR],axis=1)
df_params_SR=df_params_SR_temp.assign(eval='Param_SR')


#Tahapan deploy untuk mengaplikasikan model forecasting berdasarkan parameter terbaik yang telah dibentuk di tahahapan sebelumnya
prophet_df_SR_dict_deploy = {}
for i in list(loc_SR_df_list.keys()):
    temp_name = prophet_rename(loc_SR_df_list[i],'Tanggal','Tonase')
    temp_name = temp_name[['ds','y']]
    temp_dict = stack_norm(temp_name,'y')
    temp_date = filter_date(temp_dict, total_years = 4)

    # Build model
    model_forecast = Prophet(changepoint_prior_scale = df_params_SR[df_params_SR['Loc Src Area Name']==i]['changepoint_prior_scale'].values[0],
                     n_changepoints = df_params_SR[df_params_SR['Loc Src Area Name']==i]['n_changepoints'].values[0],
                     weekly_seasonality = True,
                     daily_seasonality = True,
                     yearly_seasonality = True,
                     interval_width = 0.95)
    model_forecast.add_country_holidays(country_name='Indonesia')
    model_forecast.fit(temp_date)
    
    # Predict
    future = model_forecast.make_future_dataframe(periods=days)
    fcst = model_forecast.predict(future)
    forecast_SR_result = fcst[['ds','yhat']]
    
    # Inverse Normalization
    yhat = pd.DataFrame({'yhat':inv_norm(forecast_SR_result['yhat'], temp_name['y'])})
    forecast_SR_result['yhat'] = yhat
    prophet_df_SR_dict_deploy[i] = forecast_SR_result


# Export Dataset
datamart_actual_SR = []
for loc_src_area_name in (loc_SR_df_list.keys()):
    temp_dm = loc_SR_df_list[loc_src_area_name]
    temp_dm['Loc Src Area Name'] = loc_src_area_name
    datamart_actual_SR.append(temp_dm)
dm_actual_SR = pd.concat(datamart_actual_SR).reset_index(drop=True)


#DM yang berisi nilai hasil forecast SR
datamart_pred_SR = []
for loc_src_area_name in (loc_SR_df_list.keys()):
    temp_dm = prophet_df_SR_dict_deploy[loc_src_area_name]
    temp_dm['Loc Src Area Name'] = loc_src_area_name
    temp_dm['Tonase'] = np.round(temp_dm['yhat'])
    datamart_pred_SR.append(temp_dm)
dm_pred_SR_temp = pd.concat(datamart_pred_SR).reset_index(drop=True)
dm_pred_SR= dm_pred_SR_temp.assign(Tipe='forecast', Jenis='forecast', Product_Type_Name='SR forecast')

# Menyusun DM_Forecasting_Mine_Ops
coal_forecast = dm_pred_coal[['ds', 'Loc Src Area Name', 'Tonase','Tipe','Jenis','Product_Type_Name']]
contractor_forecast = dm_pred_contractor[['ds', 'Contractor Name', 'Tonase','Tipe','Jenis','Product_Type_Name']]
OB_forecast = dm_pred_OB[['ds', 'Loc Src Area Name', 'Tonase','Tipe','Jenis','Product_Type_Name']]
SR_forecast = dm_pred_SR[['ds', 'Loc Src Area Name', 'Tonase','Tipe','Jenis','Product_Type_Name']]

DM_forecasting_tempA = pd.concat([coal_forecast, OB_forecast, contractor_forecast, SR_forecast], axis=0, ignore_index=True)
DM_forecasting_tempB = DM_forecasting_tempA.rename(columns={'ds': 'Tanggal','Product_Type_Name':'Product Type Name'})
#DM hasil forecasting
DM_forecasting=DM_forecasting_tempB

#DM Mine Ops
df_temp=df.copy()
df_temp['Range Calories Value']=df_temp['Range Calories Value'].astype(str)
df_temp['Calories Value']=df_temp['Calories Value'].astype(str)
DM_mine_ops= df_temp.groupby(['Tanggal','Jenis', 'Tipe', 'Contractor Name','Loc Src Area Name','Product Brand','Calories Value', 'Range Calories Value', 'Product Type Name', 'last_updated', 'Updated_Date'])['Tonase'].sum().reset_index().sort_values(by=['Tanggal'])
#df produksi batubara daily per area
DM_mine_ops['Tanggal'] = pd.to_datetime(DM_mine_ops['Tanggal'])

#Gabungan DM hasil forecast dengan mine ops
DM_forecasting_New = pd.concat([DM_mine_ops,DM_forecasting], axis=0, ignore_index=True)
start_date = pd.to_datetime('2022-01-01')
DM_forecasting_join =DM_forecasting_New[DM_forecasting_New['Tanggal'] >= start_date]

#Gabungan DM hasil forecast dengan mine ops dan Margin Error
DM_forecasting_join_final_temp = pd.concat([DM_forecasting_join, ME_Coal, ME_contractor, ME_OB, ME_SR], axis=0, ignore_index=True)
DM_forecasting_join_final_temp=DM_forecasting_join_final_temp[['Tanggal','Jenis', 'Tipe', 'Contractor Name','Loc Src Area Name','Product Brand','Calories Value', 'Range Calories Value', 'Product Type Name', 'Tonase', 'last_updated', 'Updated_Date', 'SMAPE', 'eval']].copy()


date_now = pd.datetime.now()
DM_forecasting_join_final_temp['Tanggal'] = DM_forecasting_join_final_temp['Tanggal'].fillna(date_now)
DM_forecasting_join_final_temp['last_updated'] = DM_forecasting_join_final_temp['last_updated'].fillna(date_now)
DM_forecasting_join_final_temp['Updated_Date'] = DM_forecasting_join_final_temp['Updated_Date'].fillna(date_now)

DM_forecasting_join_final=DM_forecasting_join_final_temp.fillna(0)

#Import ke DM_Forecastin_Mine_Ops
print("Sedang Import ke DM_Forecastin_Mine_Ops")
import pyodbc
from dotenv.main import load_dotenv
import os

load_dotenv()
# Koneksi ke SQL Server
server = os.environ['SERVER']
database = os.environ['DATABASE']
username = os.environ['USERNAME_1']
password = os.environ['PASSWORD']
conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)

# Melakukan operasi lainnya seperti truncate tabel dan insert data
# Truncate tabel
truncate_query = "TRUNCATE TABLE dwh.DM_Forecasting_Mine_Ops"
conn.execute(truncate_query)
cursor = conn.cursor()

# Insert ke tabel
# Loop melalui setiap baris di DataFrame dan insert ke tabel
for row in DM_forecasting_join_final.itertuples(index=False):
    insert_query = "INSERT INTO dwh.DM_Forecasting_Mine_Ops ([Tanggal], [Jenis], [Tipe], [Contractor Name], [Loc Src Area Name], " \
                   "[Product Brand], [Calories Value], [Range Calories Value], [Product Type Name], [Tonase], " \
                   "[last_updated], [Updated_Date], [SMAPE], [eval]) " \
                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    cursor.execute(insert_query, row)

# Commit perubahan
conn.commit()
# Menutup koneksi
conn.close()

print("Forecasting Done")