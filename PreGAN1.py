###################################
#PreGAN1.py
#Protein-peptide interaction region residues prediction using generative sampling technique and ensemble deep learning-based models.
#shafiee.shima@razi.ac.ir
###################################
import numpy as np
from pprint import pprint
import pandas as pd
from normal_GAN import *

def read_input_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        lines=file.read().split('\n')
        label=None
        for line in lines:
            if '>' in line:
                label=line.replace('>','')
                data[label]={'seq':'','binary':''}
            else:
                if label is not None:    
                    if '0' in line or '1' in line:
                        data[label]['binary']=line
                    else:
                        data[label]['seq']=line
    return data                        

data=read_input_file('Train.txt')
                
import concurrent.features


def process_data(label, dic):
    binary_data = dic['binary']
    seq_data = dic['seq']
    normalized_bin = call_GAN(binary_data)
    return (label, normalized_bin, seq_data)


num_threads = len(data)  

total_dic = {}
labels = []
list_of_binaries = []
list_of_seq = []
normalized_bin = []


with concurrent.features.ThreadPoolExecutor(max_workers=num_threads) as executor:
    
    feature_to_data = {executor.submit(process_data, label, dic): (label, dic) for label, dic in data.items()}
   
    for features in concurrent.features.as_completed(feature_to_data):
        label, normalized_bin_data, seq_data = features.result()
        labels.append(label)
        normalized_bin.append(normalized_bin_data)
        list_of_binaries.append(data[label]['binary'])
        list_of_seq.append(seq_data)


total_dic = {
    'label': labels,
    'binary': normalized_bin,
    'seq': list_of_seq
}

df = pd.DataFrame(total_dic)
df.to_csv('data_gan.csv', index=False)


