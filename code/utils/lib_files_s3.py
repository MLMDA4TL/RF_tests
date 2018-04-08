import boto3
import zipfile
import os
import re

import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.externals import joblib as jb


def load_sig(name_file, sum_chanels=True, timestamp=False, delimiter=';'):
    signal = np.genfromtxt(name_file, delimiter=delimiter)
    # print('Nb colonne : {0}'.format(signal.shape[1]))
    # raw files have 9 or 10 columns and np.genfromtxt adds an empty col at the
    # end so we keep only the 8 channels by removing first col and last (or last
    # two) col(s)
    if signal.shape[1] == 10:
        if timestamp:
            print("WARNING (func load_sig) : timestamp taken at first column")
            ts = np.genfromtxt(name_file, delimiter=delimiter, dtype=str)[:, 0]
            print("First column first line is : ", ts[0])
        signal = np.delete(signal, [0, 9], 1)
    elif signal.shape[1] == 11:
        if timestamp:
            ts = np.genfromtxt(name_file, delimiter=delimiter, dtype=str)[:, 9]
            print("Column num 9, first line is : ", ts[0])
        signal = np.delete(signal, [0, 9, 10], 1)
    else:
        raise IndexError('Uexpected columns number. Got {0}.'.format(signal.shape[1]))
    if sum_chanels:
        signal = np.mean(signal, 1)
    if timestamp:
        pattern = re.compile("\d\d:\d\d\.\d\d\d")
        if pattern.match(ts[0]):
            return ts, signal
        else:
            raise ValueError("ts does not match a timestamp")
    else:
        return signal


def load_sig_init(name_file, sum_chanels=True):
    signal = np.genfromtxt(name_file, delimiter=',')
    # print('Nb colonne : {0}'.format(signal.shape[1]))
    # raw files have 9 or 10 columns and np.genfromtxt adds an empty col at the
    # end so we keep only the 8 channels by removing first col and last (or last
    # two) col(s)
    if "certitude" in name_file and "1_F" in name_file: # enlever les nan  representant l'entete du fichier
        signal = signal[1:]
    if "0_FA" in name_file:
        signal = signal[1:]
    if signal.shape[1] == 9:
        print (name_file)
        signal = np.delete(signal, [0], 1)

    if sum_chanels:
        signal = np.mean(signal, 1)
    return signal


def load_sig_electrometer(name_file):
    tmp = pd.read_csv(name_file, sep=',')
    signal = np.array(tmp.loc[:, ['y_ch0']]).astype('float')
    return signal


def load_sig_electrometer_processed(name_file):
    tmp = pd.read_csv(name_file, sep=',')
    return np.array(tmp)


def get_MAC(file_mac, center_name, department_name, room_nb):
    #l = ['site', 'deviceServerAddress', 'enabled', 'protocolVersion', 'department', 'gotData', 'mac', 'debug', 'room', 'forceSyncConfig']
    # For the new MAC file Adresses
    l = ['mac', 'site', 'department', 'room']
    df = pd.read_csv(file_mac, delimiter=';', names=l, skiprows=1)
    room = df.loc[:, 'room'] == room_nb
    center = df.loc[:, 'site'] == center_name
    if department_name !="":
        department = df.loc[:, 'department'] == department_name
        idx = np.where(room & center & department)
    else:
        idx = np.where(room & center)
    if idx[0].size != 1:
        raise IndexError('MAC not found. Got {0} matches'.format(idx[0].size))
    return df.loc[idx[0][0], 'mac']


def get_center_room(file_mac, mac_ad):
    l = ['site', 'deviceServerAddress', 'enabled', 'protocolVersion', 'department', 'gotData', 'mac', 'debug', 'room', 'forceSyncConfig']
    df = pd.read_csv(file_mac, delimiter=';', names=l, skiprows=1)
    idx = np.where(df.loc[:, 'mac'] == mac_ad)
    center = df.loc[idx[0][0], 'site']
    room = int(df.loc[idx[0][0], 'room'])
    return [center, room]


def download_sig(file_mac, center_name, department_name, room_nb, date, hour, path_save='/S3_download/'):

    """
    Downloading S3 files from parameters
    """
    s3 = boto3.resource('s3')
    mac = get_MAC(file_mac, center_name, department_name, room_nb)
    file_list = []
    # files after 2017 03 21 are on debugdata-prod
    if datetime.strptime(date, '%Y.%m.%d') > datetime(2017, 3, 21):
        bucket_name = 'debugdata-prod'
    else:
        bucket_name = 'tarkett-debugdata'
    mybucket = s3.Bucket(bucket_name)
    prefix = date + '/' + mac

    # str to select right hour of s3
    # if no specified hour, all hours from the day are downloaded

    if hour == '':
        str_date_hour = ''
    else:
        str_date_hour = date.replace('.', '') + '_' + hour
        #print('str_date_hour',str_date_hour)

    file_found = False
    for obj in mybucket.objects.filter(Prefix=prefix):
        if str_date_hour in obj.key:
            file_found = True
            file_list.append(os.path.join(path_save, obj.key))
            print('File found:')
            print('{0}'.format(obj.key))
            # creating dl path
            if not os.path.exists(os.path.dirname(os.path.join(path_save, obj.key))):
                print('Creating path...')
                os.makedirs(os.path.dirname(os.path.join(path_save, obj.key)))
            elif os.path.exists(os.path.join(path_save, obj.key)):
                print('File already downloaded')
                continue
            print('Downloading file...')
            mybucket.download_file(obj.key, os.path.join(path_save, obj.key))
    if not file_found:
        print('download_sig WARNING : file not found. (hour format is hh ?)')

    return file_list


def download_folder(bucket_name, folder_name, path_save='/S3_download/',
        same_path_s3=True):
    s3 = boto3.resource('s3')
    mybucket = s3.Bucket(bucket_name)
    for obj in mybucket.objects.filter(Prefix=folder_name):
        print('File found:')
        print('{0}'.format(obj.key))
        if same_path_s3:
            file_name = obj.key
        else:
            # keeping only filename and not subdirectories of s3
            file_name = os.path.basename(obj.key)
        # creating dl path
        if not os.path.exists(os.path.dirname(os.path.join(path_save, file_name))):
            print('Creating path...')
            os.makedirs(os.path.dirname(os.path.join(path_save, file_name)))
        elif os.path.exists(os.path.join(path_save, file_name)):
            print('File already downloaded')
            continue
        print('Downloading file...')
        mybucket.download_file(obj.key, os.path.join(path_save, file_name))

    return 1


def unzip_dir(path, remove_zip=False):
    for item in os.listdir(path):  # loop through items in dir
        if item.endswith('zip'):  # check for ".zip" extension
            if os.path.exists(os.path.join(path, os.path.splitext(item)[0] +
                              '.csv')):
                # print('File already unzziped')
                continue
            else:
                print('Unzipping: ', item)
                file_name = os.path.abspath(os.path.join(path, item))  # get full path of files
                zip_ref = zipfile.ZipFile(file_name)  # create zipfile object
                zip_ref.extractall(path)  # extract file to dir
                zip_ref.close()  # close file
                if remove_zip:
                    print('Deleting zip file')
                    os.remove(file_name)  # delete zip file
    return 1

def decision_function_combination(x):
    clf_1 = jb.load("../../data/classifiers/labeled_Data/RF_Fcert1_NT50_NFEAT87_NFolds1_NRep5")
    clf_2 = jb.load("../../data/classifiers/labeled_Data/RF_Fcert2_NT50_NFEAT87_NFolds1_NRep5")

    pd_ = clf_1.predict_proba(x)[:, 1] - 0.5
    pnd = clf_2.predict_proba(x)[:, 1] - 0.5


    proba_instantane = 0.5 + 0.5 * np.amax((pd_, pnd), axis=0)
    proba_instantane = proba_instantane.ravel()

    print(np.shape(proba_instantane))
    macro_proba = pd.Series(proba_instantane).rolling(150).mean()
    print ("macro_proba", len(macro_proba))
    return proba_instantane ,macro_proba[149:]


def signal_process (sig,path_file):
    signal = pd.DataFrame()
    signal["signal preprocessed"] = pd.Series(sig)
    signal.to_csv(path_file)
    return 1

def signal_proba_macro_proba(proba,macro_proba,path_file):

    signal = pd.DataFrame()
    signal["proba_instantane"] = pd.Series(proba)
    macro_proba =list(macro_proba)
    for i in range(150):
        macro_proba.insert(0,0)
    signal["proba_macro"] = pd.Series(macro_proba)

    signal.to_csv(path_file)

    return 1
