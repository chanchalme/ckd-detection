import pandas as pd
import numpy as np
import os
import sys
import time
from collections import Counter
from sklearn.preprocessing import OneHotEncoder


def get_weighted_average(df):
    out = pd.DataFrame(columns = ['id', 'value'])
    id_list = df.id.unique().tolist()
    for id_ in id_list:
        df_sub = df[df.id.isin([id_])]
        df_sub.sort_values(["time"], inplace = True)
        avg = 0
        for i in range(1,len(df_sub)):
            avg = avg + (df_sub.iloc[i]['value'] - df_sub.iloc[i-1]['value'])/(df_sub.iloc[i]['time'] - df_sub.iloc[i-1]['time'])
        out = pd.concat([out, pd.DataFrame({'id':[id_], 'value': avg})])
    return out


def get_lab_features(data_loc, lab_df_list, df_label, train_ids, test_ids):
    '''
    Create lab data feature taking one lab_test at a time.
    Computing weighted average is handled separately
    '''
    data = pd.DataFrame(columns = ['pid','test_name','val_1st', 'val_last', 'time_1st', 'time_last', 'val_avg', 
                               'val_median', 'val_max', 'val_min'])
    
    for i,item in enumerate(lab_df_list):
        df = pd.read_csv(os.path.join(data_loc, item))
        df_weighted_average = get_weighted_average(df)
        
        test_name = item.split('.')[0].split('_')[-1]
        df['id_num'] = [i for i in range(len(df))]
        id_num_last_val = df[['id','id_num','time']].groupby('id').max('time').reset_index()['id_num'].tolist()
        id_num_first_val = df[['id','id_num','time']].groupby('id').min('time').reset_index()['id_num'].tolist()
        
        df_last_val = df[df.id_num.isin(id_num_last_val)][['id','time', 'value']]
        df_first_val = df[df.id_num.isin(id_num_first_val)][['id', 'time','value']]
    
        df_avg_val = df[['id','value']].groupby('id').mean().reset_index()
        df_median_val = df[['id','value']].groupby('id').median().reset_index()
        df_max_val = df[['id','value']].groupby('id').max().reset_index()
        df_min_val = df[['id','value']].groupby('id').min().reset_index()
        
        assert df_weighted_average.id.tolist() == df_last_val.id.tolist() == df_first_val.id.tolist() == df_avg_val.id.tolist() == df_median_val.id.tolist() == df_max_val.id.tolist() == df_min_val.id.tolist()
        
        data = pd.concat([data, pd.DataFrame(list(zip(df_last_val.id.tolist(),
                                                     [test_name]*len(df_last_val.id.tolist()),
                                                     df_first_val.value.tolist(),
                                                     df_last_val.value.tolist(),
                                                     df_first_val.time.tolist(),
                                                     df_last_val.time.tolist(),
                                                     df_avg_val.value.tolist(),
                                                     df_median_val.value.tolist(),
                                                     df_max_val.value.tolist(),
                                                     df_min_val.value.tolist(),
                                                     df_weighted_average.value.tolist()
                                                     )), 
                                             columns = ['pid','test_name','val_1st','val_last','time_1st','time_last',
                                                       'val_avg','val_median','val_max','val_min','weighted_average'])])
    data = data.merge(df_label, on='pid', how='left')
    data['last_minus_1st'] = data['val_last'] - data['val_1st']
    train_data = data[data.pid.isin(train_ids)]
    test_data = data[data.pid.isin(test_ids)]
    return train_data, test_data


def get_lab_array(data, feature_list, feature_header = []):
    data.sort_values(["test_name","pid"], inplace = True)
    np_list = []
    for test in data.test_name.unique().tolist():
        test_list = [i + '_{}'.format(test) for i in feature_list]
        np_list.append(np.array(data[data.test_name.isin([test])][feature_list]))
        feature_header.extend(test_list)
    X = np.hstack(np_list)
    print(X.shape)
    l1 = []
    data1 = data[['pid','Stage_Progress']]
    data1.drop_duplicates(inplace=True)
    data1.sort_values(["pid"], inplace = True)
    
    for i in data1['Stage_Progress'].tolist():
        l1.append(1) if i else l1.append(0)
    Y = np.array(l1)
    pid_arr = np.array(data1.pid.tolist())
    print(np.unique(Y, return_counts=True))
    return X, Y, pid_arr, feature_header

def encode_one_hot(df, index_key='pid', col_name = 'race'):
    enc = OneHotEncoder(handle_unknown='error')
    df.sort_values([index_key], inplace = True)
    
    X = np.array(df[col_name].unique().tolist()).reshape(-1, 1)
    enc.fit(X)
    col_list = enc.categories_[0].tolist()
    race_one_hot = enc.transform(np.array(df[col_name].tolist()).reshape(-1,1)).toarray()
    out = pd.DataFrame(race_one_hot, columns = col_list)
    out[index_key] = df[index_key].tolist()
    return out

def get_binary(df, index_key='pid', col_name='gender', pos_val='Male'):
    df = df[[index_key, col_name]]
    out_list = []
    for _, row in df.iterrows():
        if row[col_name].strip() == pos_val:
            out_list.append(1)
        else: 
            out_list.append(0)
    df[col_name] = out_list
    return df


def get_demo_features(df, train_ids, test_ids):
    df_race = encode_one_hot(df)
    df_gender = get_binary(df)
    df_age = df[['pid','age']]
    out = df_age.merge(df_gender, how = 'left', on = 'pid')
    out = out.merge(df_race, how = 'left', on = 'pid')    
    out_train = out[out.pid.isin(train_ids)]
    out_test = out[out.pid.isin(test_ids)]
    return out_train, out_test

def get_df_array(df):
    df.sort_values(["pid"], inplace = True)    
    pid_arr = np.array(df.pid.tolist())
    df.drop(columns=['pid'],inplace=True)
    col_names = np.array(df.columns.tolist())
    X = np.array(df)
    return X, col_names, pid_arr
    
def get_meds_feature(df_meds, pid_list, train_ids, test_ids):
    df_meds['dose_duration'] = df_meds['end_day'] - df_meds['start_day']
    df_meds.rename(columns = {'id':'pid'}, inplace=True)
    drug_list = df_meds.drug.unique().tolist()
    filter_drugs = ['atenolol', 'atorvastatin', 'carvedilol', 'losartan', 'lovastatin', 'metformin', 'rosuvastatin', 'simvastatin']
    
#     medication - indicator feature
    df_meds_ind = df_meds[['pid', 'drug']].groupby(['pid', 'drug']).count().reset_index()
    df_meds_ind['ind'] = 1
    for pid in pid_list:
        for drug_name in drug_list:
            if drug_name not in df_meds_ind[df_meds_ind['pid'] == pid].drug.unique().tolist():
                temp_dict = {'pid': pid, 'drug': drug_name, 'ind': 0}
                df_meds_ind = df_meds_ind.append(temp_dict, ignore_index=True)
    df_meds_ind1 = df_meds_ind.pivot_table(index = 'pid', columns = 'drug', values = 'ind')
    df_meds_ind1 = df_meds_ind1[filter_drugs]
    drug_ind_col_list = [str(x) + '_ind' for x in df_meds_ind1.columns]
    df_meds_ind1.columns = drug_ind_col_list
    df_meds_ind1.reset_index(inplace=True)
    
#     # medication - average dosage duration feature
#     df_meds_last_date = df_meds[['pid', 'drug', 'end_day']].groupby(['pid', 'drug']).max().reset_index()
#     for pid in pid_list:
#         for drug_name in drug_list:
#             if drug_name not in df_meds_last_date[df_meds_last_date['pid'] == pid].drug.unique().tolist():
#                 temp_dict = {'pid': pid, 'drug': drug_name, 'end_day': 0}
#                 df_meds_last_date = df_meds_last_date.append(temp_dict, ignore_index=True)
#     #import pdb; pdb.set_trace()
#     df_meds_last_date1 = df_meds_last_date.pivot_table(index = 'pid', columns = 'drug', values = 'end_day')
#     df_meds_last_date1 = df_meds_last_date1[filter_drugs]
#     meds_last_col_list = [str(x) + '_last_dose' for x in df_meds_last_date1.columns]
#     df_meds_last_date1.columns = meds_last_col_list
#     df_meds_last_date1.reset_index(inplace=True)
#     print(meds_last_col_list)
    
    df_meds_train = df_meds_ind1[df_meds_ind1.pid.isin(train_ids)]
    df_meds_test = df_meds_ind1[df_meds_ind1.pid.isin(test_ids)]

#     df_meds_train = df_meds_last_date1[df_meds_last_date1.pid.isin(train_ids)]
#     df_meds_test = df_meds_last_date1[df_meds_last_date1.pid.isin(test_ids)]
    
    return df_meds_train, df_meds_test

