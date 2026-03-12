import torch
import torch.nn as nn

# HPO best config = h1=512, h2=128, h3=32, h4=16, lr=1e-05
# original config = h1=512, h2=256, h3=128, h4=64, output=2

## Model used : 3 layers
class fraudmodel_3layer(nn.Module):
    def __init__(self, input_size, hidden_layers=[64, 16], output=2):
        super(fraudmodel_3layer, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.relu1 = nn.LeakyReLU() 
        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.dropout1 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.relu2 = nn.LeakyReLU() 
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(hidden_layers[1], output)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        return out


## Model used : 5 layers
class fraudmodel_5layer(nn.Module):
    def __init__(self, input_size=282, hidden_layers=[512, 256, 128, 64], output=2):
        super(fraudmodel_5layer, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(hidden_layers[0], eps=1e-5)
        self.dropout1 = nn.Dropout(p=0.20)

        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.relu2 = nn.LeakyReLU() 
        self.bn2 = nn.BatchNorm1d(hidden_layers[1], eps=1e-5)
        self.dropout2 = nn.Dropout(p=0.20)

        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.relu3 = nn.LeakyReLU() 
        self.bn3 = nn.BatchNorm1d(hidden_layers[2], eps=1e-5)
        self.dropout3 = nn.Dropout(p=0.20)

        self.fc4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.relu4 = nn.LeakyReLU() 
        self.bn4 = nn.BatchNorm1d(hidden_layers[3], eps=1e-5)
        self.dropout4 = nn.Dropout(p=0.20)

        self.fc5 = nn.Linear(hidden_layers[3], output)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu3(out)
        out = self.bn3(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.bn4(out)
        out = self.dropout4(out)

        out = self.fc5(out)

        return out
    
    

class fraudmodel_7layer(nn.Module):
    def __init__(self, input_size=255, hidden_layers=[128, 64, 32, 16, 8, 4], output=2):
        super(fraudmodel_7layer, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.relu1 = nn.LeakyReLU() 
        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.dropout1 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.relu2 = nn.LeakyReLU() 
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.dropout2 = nn.Dropout(p=0.25)

        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.relu3 = nn.LeakyReLU() 
        self.bn3 = nn.BatchNorm1d(hidden_layers[2])
        self.dropout3 = nn.Dropout(p=0.25)

        self.fc4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.relu4 = nn.LeakyReLU() 
        self.bn4 = nn.BatchNorm1d(hidden_layers[3])
        self.dropout4 = nn.Dropout(p=0.25)

        self.fc5 = nn.Linear(hidden_layers[3], hidden_layers[4])
        self.relu5 = nn.LeakyReLU() 
        self.bn5 = nn.BatchNorm1d(hidden_layers[4])
        self.dropout5 = nn.Dropout(p=0.25)

        self.fc6 = nn.Linear(hidden_layers[4], hidden_layers[5])
        self.relu6 = nn.LeakyReLU() 
        self.bn6 = nn.BatchNorm1d(hidden_layers[5])
        self.dropout6 = nn.Dropout(p=0.25)

        self.outp = nn.Linear(hidden_layers[5], output)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu3(out)
        out = self.bn3(out)
        out = self.dropout3(out)

        out = self.fc4(out)
        out = self.relu4(out)
        out = self.bn4(out)
        out = self.dropout4(out)

        out = self.fc5(out)
        out = self.relu5(out)
        out = self.bn5(out)
        out = self.dropout5(out)

        out = self.fc6(out)
        out = self.relu6(out)
        out = self.bn6(out)
        out = self.dropout6(out)

        out = self.outp(out)

        return out

    
    
class fraudmodel_8layer(nn.Module):   ## 8 layers
    def __init__(self, input_size=255, hidden_layers=[512, 128, 64, 32, 16, 8, 4], output=2):
        super(fraudmodel_8layer, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.relu1 = nn.LeakyReLU() 
        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.dropout1 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.relu2 = nn.LeakyReLU() 
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.dropout2 = nn.Dropout(p=0.25)

        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.relu3 = nn.LeakyReLU() 
        self.bn3 = nn.BatchNorm1d(hidden_layers[2])
        self.dropout3 = nn.Dropout(p=0.25)

        self.fc4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.relu4 = nn.LeakyReLU() 
        self.bn4 = nn.BatchNorm1d(hidden_layers[3])
        self.dropout4 = nn.Dropout(p=0.25)

        self.fc5 = nn.Linear(hidden_layers[3], hidden_layers[4])
        self.relu5 = nn.LeakyReLU() 
        self.bn5 = nn.BatchNorm1d(hidden_layers[4])
        self.dropout5 = nn.Dropout(p=0.25)

        self.fc6 = nn.Linear(hidden_layers[4], hidden_layers[5])
        self.relu6 = nn.LeakyReLU() 
        self.bn6 = nn.BatchNorm1d(hidden_layers[5])
        self.dropout6 = nn.Dropout(p=0.25)

        self.fc7 = nn.Linear(hidden_layers[5], hidden_layers[6])
        self.relu7 = nn.LeakyReLU() 
        self.bn7 = nn.BatchNorm1d(hidden_layers[6])
        self.dropout7 = nn.Dropout(p=0.25)

        self.outp = nn.Linear(hidden_layers[6], output)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu3(out)
        out = self.bn3(out)
        out = self.dropout3(out)

        out = self.fc4(out)
        out = self.relu4(out)
        out = self.bn4(out)
        out = self.dropout4(out)

        out = self.fc5(out)
        out = self.relu5(out)
        out = self.bn5(out)
        out = self.dropout5(out)

        out = self.fc6(out)
        out = self.relu6(out)
        out = self.bn6(out)
        out = self.dropout6(out)

        out = self.fc7(out)
        out = self.relu7(out)
        out = self.bn7(out)
        out = self.dropout7(out)

        out = self.outp(out)

        return out


def get_compression_ratio(teacher_model, student_model):
    """Calculate compression ratio between teacher and student models."""
    teacher_params = get_model_parameters_count(teacher_model)
    student_params = get_model_parameters_count(student_model)
    return teacher_params / student_params