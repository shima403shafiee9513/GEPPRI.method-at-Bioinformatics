###################################
#PreGAN2.py
#Protein-peptide interaction region residues prediction using generative sampling technique and ensemble deep learning-based models.
#shafiee.shima@razi.ac.ir
###################################

import pandas as pd


def read_csv_file(file_path):
    data = {}
    df = pd.read_csv(file_path)
    labels = df['label']
    binaries = df['binary']
    seqs = df['seq']
    for label, binary, seq in zip(labels, binaries, seqs):
        data[label] = {'binary': binary, 'seq': seq}
    return data




import numpy as np

def preprocess_binary_data(binary_data):
  
    two_dimensional_array = []
    window_size=7
    
    for i in range(0, len(binary_data), window_size):
        
        row = []
        
        for j in range(window_size):
            if i + j < len(binary_data):
                row.append(int(binary_data[i + j]))
            else:
                
                row.append(0)
        
        two_dimensional_array.append(row)
        
    if len(two_dimensional_array)<18:
        for i in range(len(two_dimensional_array),18):
            row=[0]*7
            two_dimensional_array.append(row)
    elif len(two_dimensional_array)>=18:
              two_dimensional_array=two_dimensional_array[0:18]  
    return  two_dimensional_array

def preprocess_seq_data(seq_data):
   
    two_dimensional_array = []
    window_size=7
    
    for i in range(0, len(seq_data), window_size):
        
        row = []
        
        for j in range(window_size):
            if i + j < len(seq_data):
                row.append(str(seq_data[i + j]))
            else:
                
                row.append('_')
        
        two_dimensional_array.append(row)
        
    if len(two_dimensional_array)<18:
        for i in range(len(two_dimensional_array),18):
            row=["_"]*7
            two_dimensional_array.append(row)
    elif len(two_dimensional_array)>=18:
              two_dimensional_array=two_dimensional_array[0:18]  
    return  two_dimensional_array                  

def main():
    
    file_path = 'data_Gan.csv'
    
    data = read_csv_file(file_path)
    dic_final={}
    
    for label, info in data.items():
        
        bin_matrix=preprocess_binary_data(info['binary'])
        seq_matrix=preprocess_seq_data(str(info['seq']))
        dic_final[label]={'binary':bin_matrix,'seq':seq_matrix}
 
    return dic_final

if __name__=="__main__":
    dic_final=main()
    print(dic_final)        

        

