###################################
#Feature groups.py
#Protein-peptide interaction region residues prediction using generative sampling technique and ensemble deep learning-based models.
#shafiee.shima@razi.ac.ir
###################################
import numpy as np
import pandas as pd
import sys
import string
import math
import os
import numpy
import scipy
import scipy.io
import cPickle
import pickle

#Function

def compute_entropy(dis_list):
    """compute shannon entropy for a distribution.
    base = len(dis_list) is the base of log function 
    to make entropy between 0 and 1."""
    
    if sum(dis_list) == 0:
        return 0.0
    prob_list = map(lambda x:(x+0.0)/sum(dis_list),dis_list)
    ent = 0.0
    for prob in prob_list:
        if prob != 0:
            ent -= prob*math.log(prob,len(dis_list))
    return ent

*****
fin = file(pssm_path+pid+'.pssm','r')
pssm = fin.readlines()
fin.close()
if len(pssm[-6].split()) != 0 or pssm[3].split()[0] != '1': 
    print 'error on reading pssm, line -6 is not a spare line;\
     or line 3 is not the first line'
    sys.exit(1)
pssm = pssm[3:-6]
pssm_res = map(lambda x:x.split()[1],pssm)

fin = file(dssp_path+pid+'.pdb.txt','r')
dssp = fin.readlines()[1:] 
fin.close()

cnt_dssp = 0
for line in dssp:
	cnt_dssp += 1
	if '#' in line:
		strt_pnt = cnt_dssp
dssp = dssp[strt_pnt:]
*****
fin = file(rsa_path+pid+'.spd2','r')
sp = fin.readlines()[1:]
fin.close()
rsa_res = ''.join([x.split()[1] for x in sp])
rsa_pre = [string.atof(x.split()[6]) for x in sp]
fin = file(pssm_path+pid+'.pssm','r')
pssm = fin.readlines()
fin.close()
if len(pssm[-6].split()) != 0 or pssm[3].split()[0] != '1': 
    print 'error on reading pssm, line -6 is not a spare line;\
     or line 3 is not the first line'
    sys.exit(1)
pssm = pssm[3:-6]
fin = file(fasta_path+pid+'.seq','r')
ann = fin.readlines()
fin.close()
if len(ann) != 2:
    print 'check sequence',pid
    sys.exit(1)
fastaseq = ann[1].split()[0]
if not fastaseq == rsa_res:
    print 'Sequence inconsistent!'
    print 'fasta: ',fastaseq
    print '  rsa: ',rsa_avg
    exit(1)
fout = file(fea_path+pid+'.info','w')
fout.write('>%s\n' %pid)
pos = 0
for i in xrange(len(fastaseq)):
    res = fastaseq[i]
    fout.write('%5d%5s%5s'%(i+1,res,res))
    if pssm[pos].split()[1] == res:# check for residue type
        for p_e in pssm[pos].split()[2:22]:
            fout.write(':%2s' %p_e)
        for p_e in pssm[pos].split()[22:42]:
            fout.write(':%3s' %p_e)
        fout.write(':%5s' %pssm[pos].split()[42])
    else:
        print 'Error reading pssm file!'
        flog = file(error_file,'a')
        flog.write(pid+': error on writing pssm, %s:%s\n' \
        %(pssm[pos].split()[1],res))
        flog.close()
        sys.exit(1)
    if rsa_res[pos] == res:
        fout.write(':%5.3f' %rsa_pre[pos])
    else:
        print 'Error reading rsa file!'
        flog = file(error_file,'a')
        flog.write(pid+': error on writing rsa, %s:%s\n' %(rsa_res[pos],res))
        flog.close()
        sys.exit(1)
    pos += 1
    fout.write('\n')
fout.close()
#---------------------------------------------------
pdb_residue_temp = []
pdb_residue = []
all_coord = []
all_X = []
all_Y = []
all_Z = []
b_factor = []
pdbfile = []
all_residue_number = []
residue_number = []
#Angles are {θ, τ, φ, and ψ}
with open('your_pdb_directory'+pid+'.pdb') as pdb_file:
	for line in pdb_file:
		if line[:4] == 'ATOM' or line[:6] == "HETATM" and line:
			temp_pdbfile = ('%s%5.2f'%(line[:55].rstrip(),1.00))
			pdbfile.append(temp_pdbfile)
			all_residue_number.append(line[22:26])
			if line[12:16] == ' CA ':
				residue_number.append(line[22:26])
				pdb_residue_temp.append(line[17:20])
				all_coord.append([line[30:38], line[38:46], line[46:54]])
				all_X.append(line[30:38])
				all_Y.append(line[38:46])
				all_Z.append(line[46:54])
				b_factor.append(line[57:60])
all_residue_number = [x.replace(' ', '') for x in all_residue_number]
for i in xrange(len(pdb_residue_temp)):
	pdb_residue.append(ext_pdb_residue[pdb_residue_temp[i]])
	
#---------------------------------------------------
for i in xrange(win):
    wop.insert(0,'%7.5f' %(0))
    wop.append('%7.5f' %(0))
#PP = {steric parameters, polarizability, helix probability, hydrophobicity, volume, isoelectric point, sheet probability}
for i in xrange(P7_win,seq_len):
    for j in xrange(i-P7_win,i+P7):
        out_list[i-P7_win].append(PP[j])
#---------------------------------------------------
fin = file(HSE_path+pid+'.pdb.txt','r')
aa = fin.readlines()[1:] 
fin.close()
HSE_res = map(lambda x:x.split()[3],aa)
act_CN = map(lambda x:x.split()[4],aa)
act_HSE_UP = map(lambda x:x.split()[5],aa)
act_HSE_DN = map(lambda x:x.split()[6],aa)
act_CN = [string.atof(x.strip(' ')) for x in act_CN]
min_cn = min(act_CN)
max_cn = max(act_CN)
CN = [((q - min_cn)/(max_cn-min_cn)) for q in act_CN]
act_HSE_UP = [string.atof(x.strip(' ')) for x in act_HSE_UP]
min_up = min(act_HSE_UP)
max_up = max(act_HSE_UP)
HSE_UP = [((q - min_up)/(max_up-min_up)) for q in act_HSE_UP]
act_HSE_DN = [string.atof(x.strip(' ')) for x in act_HSE_DN]
min_dn = min(act_HSE_DN)
max_dn = max(act_HSE_DN)
HSE_DN = [((q - min_dn)/(max_dn-min_dn)) for q in act_HSE_DN]
#---------------------------------------------------
def encode_restype(res):
    """binary encoding residue type."""
    AAs = 'ARNDCQEGHILKMFPSTWYV'
    code = []
    for a in AAs:
        if res == a:
            code.append('1')
        else:
            code.append('0')

              
    return code
    
#in case of Test.Set True in case of Train.Set False
#Train file includes Train1 or Train2
#Test file contains Test1 or Test2
isTest = False
if is Test:
    filename = "Test.Set.txt"
else:#Train
    filename = "Train.Set.txt"

# Read file in a numpy array
mat = np.loadtxt(filename,dtype=str)

if isTest:
 
    output = "Test.Set.txt"
else:
   
    output = "Train.Set.txt"
with open(output,"w+") as out:
    head = 'name	no	AA	  Phi   Psi	 Theta(i-1=>i+1)	Tau(i-2=>i+1)	 A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V	P1	P2	P3	P4	P5	P6	P7	 HSE_A_U  	HSE_A_D    HSE_B_U   HSE_B_D  CNCC-0   AAO.0  AAO.1 Label'  
    np.savetxt(out,mat,delimiter='\t',header=head,fmt='%s')
#---------------------------------------------------	
#The tools introduced in the following sources are used:

1.Taherzadeh, G., Zhou, Y., Liew, A. W. C., & Yang, Y. (2018). Structure-based prediction of protein–peptide binding regions using Random Forest. Bioinformatics, 34(3), 477-484.

2.Taherzadeh, G., Yang, Y., Zhang, T., Liew, A. W. C., & Zhou, Y. (2016). Sequence-based prediction of protein–peptide binding sites using support vector machine. Journal of computational chemistry, 37(13), 1223-1229. 

3.Altschul, S. F., Madden, T. L., Schäffer, A. A., Zhang, J., Zhang, Z., Miller, W., & Lipman, D. J. (1997). Gapped BLAST and PSI-BLAST: a new generation of protein database search programs. Nucleic acids research, 25(17), 3389-3402.

4.Heffernan, R., Yang, Y., Paliwal, K., & Zhou, Y. (2017). Capturing non-local interactions by long short-term memory bidirectional recurrent neural networks for improving prediction of protein secondary structure, backbone angles, contact numbers and solvent accessibility. Bioinformatics, 33(18), 2842-2849. 

5.Islam, M. N., Iqbal, S., Katebi, A. R., & Hoque, M. T. (2016). A balanced secondary structure predictor. Journal of theoretical biology, 389, 60-71.

6.Heffernan, R., Dehzangi, A., Lyons, J., Paliwal, K., Sharma, A., Wang, J., ... & Yang, Y. (2016). Highly accurate sequence-based prediction of half-sphere exposures of amino acid residues in proteins. Bioinformatics, 32(6), 843-849.

#---------------------------------------------------