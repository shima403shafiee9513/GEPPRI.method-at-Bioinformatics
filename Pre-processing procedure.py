###################################
#Pre-processing procedure.py
#Protein-peptide interaction region residues prediction using generative sampling technique and ensemble deep learning-based models.
#shafiee.shima@razi.ac.ir
###################################
"import numpy as np\n",
    "from sklearn.preprocessing import MinMaxScaler, StandardScaler\n",
    "\n",
    "normal_type=0 # 0 is MinMax, 1 is StandardScaler\n",
    "\n",
    "#------------- clean---------------------\n",
    "# remove some columns of train and test dataset\n",
    "# train\n",
    "lbl_train=np.array(train_pd[\"Label\"],dtype=int)\n",
    "train_pre=train_pd.drop(columns=[\"# name\",\"no\",\"AA\",\"Label\"])\n",
    "# test\n",
    "lbl_test=np.array(test_pd[\"Label\"],dtype=int)\n",
    "test_pre=test_pd.drop(columns=[\"# name\",\"no\",\"AA\",\"Label\"])\n",
    "\n",
    "#------------- transform-----------------\n",
    "from sklearn import pre-processing\n",
    "le = pre-processing.LabelEncoder()\n",
    "# convert string to int\n",
    "col_num=le.fit(train_pre[\"SS\"])\n",
    "train_pre[\"SS\"]=le.transform(train_pre[\"SS\"])\n",
    "test_pre[\"SS\"]=le.transform(test_pre[\"SS\"])\n",
    "#------------- drop and fill nan------------\n",
    "#***train\n",
    "indices_to_keep = ~train_pre.isin([np.nan, np.inf, -np.inf]).any(axis=1)\n",
    "train_pre=train_pre[indices_to_keep].astype(np.float64)\n",
    "lbl_train=lbl_train[indices_to_keep].astype(np.float64)\n",
    "#***test\n",
    "indices_to_keep = ~test_pre.isin([np.nan, np.inf, -np.inf]).any(axis=1)\n",
    "test_pre=test_pre[indices_to_keep].astype(np.float64)\n",
    "lbl_test=lbl_test[indices_to_keep].astype(np.float64)\n",
    "train_pre=train_pre.fillna(0)\n",
    "test_pre=test_pre.fillna(0)\n",
    "data_train=np.array(train_pre,dtype=float)\n",
    "data_test=np.array(test_pre,dtype=float)\n",
    "\n",
    "#------------- join train and test--------\n",
    "Data=np.concatenate((data_train,data_test),axis=0)\n",
    "lbl=np.concatenate((lbl_train,lbl_test),axis=0)\n",
    "\n",
    "#---------------normalization ------------\n",
    "method_norm= MinMaxScaler(feature_range=(0,1))\n",
    "if(normal_type==1):\n",
    "  method_norm=StandardScaler()\n",
    "Data_normal=method_norm.fit_transform(Data)\n",
    "data_train_normal=method_norm.transform(data_train)\n",
    "data_test_normal=method_norm.transform(data_test)\n",
    "\n",
    "#------------- show----------------------\n",
    "print(\"-\"*10)\n",
    "print(\"number of sample: \",Data_normal.shape[0] )\n",
    "print(\"number of features: \",Data_normal.shape[1] )\n",
    "print(\"number of class one: \",np.sum(lbl==0) )\n",
    "print(\"number of class two: \",np.sum(lbl==1)  )"
   ]
  },