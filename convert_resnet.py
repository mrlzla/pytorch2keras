import torch.nn as nn
import torch
import numpy as np
from keras.layers import Activation
from keras.models import Model
from torch.autograd import Variable
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet34
from pytorch2keras.converter import pytorch_to_keras as p2k

input_np = np.random.uniform(0, 1, (1, 3, 512, 512))
input_var = Variable(torch.FloatTensor(input_np))


for model_fn in [resnet18, resnet34]:
    model_name = model_fn.__name__
    model = model_fn(True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    output = model(input_var)
    k_model = p2k((3, 512, 512), output)
    #x = k_model.output
    #x = Activation('softmax')(x)
    #k_model = Model(k_model.input, x)
    k_output = k_model.predict(input_np.transpose([0, 2, 3, 1]))
    #k_model.save("{}_imagenet.h5".format(model_name))
    print("{} is converted".format(model_name))
    #print(k_model.summary())
    print(k_output)
    print(output.data.cpu().numpy())
    #print(np.fabs(k_output - output.data.cpu().numpy()))
