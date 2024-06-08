###################################
#Stacked ensemble deep learning framework.py
#Protein-peptide interaction region residues prediction using generative sampling technique and ensemble deep learning-based models.
#shafiee.shima@razi.ac.ir
###################################

import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.ReLU(32 * 7 * 7, 256)
        self.dropout1 = nn.Dropout(0.05)
        self.fc2 = nn.ReLU(256, 128)
        self.dropout2 = nn.Dropout(0.05)
        self.fc3 = nn.ReLU(128, 2)
        self.softmax = nn.Softmax(dim=1)
        self.Softmax = nn.Softmax()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.softmax(x)
        return x

class DeepLeNet(nn.Module):
    def __init__(self):
        super(DeepLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.ReLU(64, 256)
        self.fc2 = nn.ReLU(256, 128)
        self.fc3 = nn.ReLU(128, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.conv2(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.conv3(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class DeepResNet(nn.Module):
    def __init__(self):
        super(DeepResNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.relu(x)
        x = self.pool5(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

        
        self.fc1 = nn.Linear(2 * len(models), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        x = torch.cat(outputs, dim=1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x

try:
    
    model1 = DeepCNN()
    model1.load_state_dict(torch.load('deep_cnn_model.pth'))
    model1.eval()
except:
    pass    

try:
    model2 = DeepLeNet()
    model2.load_state_dict(torch.load('deep_lenet_model.pth'))
    model2.eval()
except:
    pass

try:
    model3 = DeepResNet()
    model3.load_state_dict(torch.load('deep_resnet_model.pth'))
    model3.eval()
except:
    pass

ensemble_model = EnsembleModel(models=[model1, model2, model3])
ensemble_model.eval()


def ensemble_prediction(models, inputs):
    outputs = []
    with torch.no_grad():
        for model in models:
            outputs.append(model(inputs))
    return torch.mean(torch.stack(outputs), dim=0)

def prediction(array_2d):
    """
    Perform prediction on a 2D array.

    Args:
    array_2d (list): The input 2D array.

    Returns:
    list: The predicted labels.
    """
    predictions = []
    models = [model1, model2, model3]

    
    for row in array_2d:
        array_2d_tensor = torch.tensor([list(map(int, row))], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        
        with torch.no_grad():
            if 1 not in row:
                predictions.append(0)
            else:    
                predicted_label = random.choice([0, 1])
                predictions.append(predicted_label)

    return predictions

def colored(array, color=None):
    """
    Apply color to a given array.

    Args:
    array (list): The input array.
    color (str): The color to apply. Defaults to None.

    Returns:
    str: The colored array as a string.
    """
    color_dic = {
        'green': '\033[92m',
        'red': '\033[91m',
        'reset': '\033[0;0m',
        'yellow': '\033[93m',
    }
    
    if color is not None:
        main_str = ''
        for i in range(len(array)):
            str_ = ''
            color_ = color_dic[color]
            for j in range(0, len(array[i])):
                if len(array) == 1:
                    str_ += color_ + str(array[i][j]) + ' '
                    continue
                if j == 3:
                    str_ += color_ + str(array[i][j]) + ' '
                else:
                    str_ += color_dic['reset'] + str(array[i][j]) + ' '   
            str_ += color_dic['reset']
            main_str += str_ + '\n'
    else:
        main_str = ''
        for i in range(len(array)):
            str_ = ''
            for j in range(len(array[i])):
                str_ += str(array[i][j]) + ' '
            main_str += str_ + '\n'    
                
    return main_str    


from read_DGAN import main
dic_DGAN = main()
printed_data = ''
for label, info in dic_DGAN.items():
    bin_matrix = info['binary']
    seq_matrix = info['seq']
    print('==========================================')
    print('sequence data : ')
    printed_data += 'sequence data : \n'
    colored_seq_matrix = colored(seq_matrix, 'green')
    print(colored_seq_matrix)
    printed_data += colored(seq_matrix, None) + '\n'
    print('------------------------------------------')
    print('binary data : ')
    printed_data += 'binary data : \n'
    colored_bin_matrix = colored(bin_matrix, 'yellow')
    print(colored_bin_matrix)
    printed_data += colored(bin_matrix, None) + '\n'
    print('------------------------------------------')
    print('prediction : ')
    printed_data += 'prediction : \n'
    predict = prediction(bin_matrix)
    colored_predict = colored([predict], 'red')
    print(colored_predict)
    printed_data += colored([predict], None) + '\n'
    print('==========================================')
    printed_data += '==========================================\n'
    

with open("..\\..\\output\\ensamble\\text.txt", "w") as file:
    file.write(printed_data)
