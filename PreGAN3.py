###################################
#PreGAN3.py
#Protein-peptide interaction region residues prediction using generative sampling technique and ensemble deep learning-based models.
#shafiee.shima@razi.ac.ir
###################################

import numpy as np
import pandas as pd
#test
filename = "Test.txt"
#train
#filename = "Train.txt"

# Read file in a numpy array
mat = np.loadtxt(filename,dtype=str)
print(mat.shape)

#Add a column for window
mat = np.hstack((mat, np.empty((mat.shape[0], 1))))
print(mat[0])

#find the first index of each protein
proteins,index= np.unique(mat[:,0],return_index=True)
index = np.append(index,len(mat))

for i,p in enumerate(proteins):
    start = index[i]
    end = index[i+1]
    mat[start,42]=mat[start+3,2]+mat[start+2,2]+mat[start+1,2]+mat[start,2]+mat[start+1,2]+mat[start+2,2]+mat[start+3,2]
    mat[start+1,42]=mat[start+2,2]+mat[start+1,2]+mat[start,2]+mat[start+1,2]+mat[start+2,2]+mat[start+3,2]+mat[start+4,2]
    mat[start+2,42]=mat[start+1,2]+mat[start,2]+mat[start+1,2]+mat[start+2,2]+mat[start+3,2]+mat[start+4,2]+mat[start+5,2]
    for j in range(start+3,end-3):
        mat[j,42]=mat[j-3,2]+mat[j-2,2]+mat[j-1,2]+mat[j,2]+mat[j+1,2]+mat[j+2,2]+mat[j+3,2]
    mat[end-3,42]=mat[end-6,2]+mat[end-5,2]+mat[end-4,2]+mat[end-3,2]+mat[end-2,2]+mat[end-1,2]+mat[end-2,2]
    mat[end-2,42]=mat[end-5,2]+mat[end-4,2]+mat[end-3,2]+mat[end-2,2]+mat[end-1,2]+mat[end-2,2]+mat[end-3,2]
    mat[end-1,42]=mat[end-4,2]+mat[end-3,2]+mat[end-2,2]+mat[end-1,2]+mat[end-2,2]+mat[end-3,2]+mat[end-4,2]

df = pd.DataFrame(mat, columns = ['name','no','AA','Phi','Psi','Theta(i-1=>i+1)','Tau(i-2=>i+1)','A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','P1','P2','P3','P4','P5','P6','P7','HSE_A_U','HSE_A_D','HSE_B_U','HSE_B_D','CNCC-0','AAO.0','AAO.1','Window'])

print(df[df['name'] == '1avpA'])
print(df[df['name'] == '1avpA'][['no','AA','Window']])

#test output
output = "Test.txt"
#train output
#output = "Train.txt"
with open(output,"w+") as out:
    head = 'name no	AA	Phi	Psi	Theta(i-1=>i+1)	Tau(i-2=>i+1)	A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V	P1	P2	P3	P4	P5	P6	P7	HSE_A_U  	HSE_A_D    HSE_B_U   HSE_B_D  CNCC-0   AAO.0  AAO.1 Window'
    np.savetxt(out,mat,delimiter='\t',header=head,fmt='%s')


