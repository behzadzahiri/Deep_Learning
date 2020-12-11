import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os


file = os.path.abspath(__file__)
datadir = os.path.dirname(file)

movies = pd.read_csv(datadir + r'\ml-1m\movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv(datadir + r'\ml-1m\users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv(datadir + r'\ml-1m\ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')




# Preparing the training set and the test set
training_set = pd.read_csv(datadir + r'\ml-100k\u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv(datadir + r'\ml-100k\u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))





#  Converting the data into an array with users in lines and movies in columns
# this function basically converts the dataframe into a huge list of lists, where each sublist is the ratings of each user
# for those movies that are not rated, we set them to 0. Pay attention to the array of np.zeros. We create an array of all-zero,
# and then update the values based on index
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies  = data[:, 1][data[:,0] == id_users]
        id_ratings = data[:, 2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        #we need a list of list in the end, because this is the format Torch expects
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)


# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)



# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh): #nv is number of visible nodes; nh is number of hidden nodes
        self.W = torch.randn(nh, nv) #parameter for weights
        self.a = torch.randn(1, nh) #parameter for hidden nodes
        self.b = torch.randn(1, nv) #parameter for visible nodes
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t()) #compute the product of weights and neurons
        activation = wx + self.a.expand_as(wx) #activation function which is weights plus the bias
        p_h_given_v = torch.sigmoid(activation) #probability that the hidden node is activated given the value of the visible node
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W) #compute the product of weights and neurons
        activation = wy + self.b.expand_as(wy) #activation function which is weights plus the bias
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h) #probability that the hidden node is activated given the value of the visible node
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)


nv = len(training_set[0]) #number of movies (visible nodes)
nh = 100 #number of hidden nodes: can e chosen any number
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10 #number of epochs
for epoch in range(1, nb_epoch + 1):
    train_loss = 0 #we're going to normalize this train loss
    s = 0. #counter
    for id_user in range(0, nb_users - batch_size, batch_size): #we're getting each batch and within each apply the functions.
    #note that we don't want to update the weights after each user, but rather update them after each series of users (batch) gets into the network
        vk = training_set[id_user:id_user+batch_size] #ratings of users in a batch: so this for instance gives of the batch of 100 users
        v0 = training_set[id_user:id_user+batch_size] #target
        ph0,_ = rbm.sample_h(v0) #initial probabilities: probability that the hidden node at the start equal to 1, given the real ratings. We take Sample_h, because it returns p_h_given_v
        #the ,_ addition is for getting the first element of the return function (in this case, the sample_h function returns 2 elements)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk) #update vk; sampled visible nodes after the first step of Gibb sampling
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))































