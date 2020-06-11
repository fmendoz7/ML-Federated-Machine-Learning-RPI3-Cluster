#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install PySyft in Google Colab

get_ipython().system('pip install tf-encrypted==0.5.6')
get_ipython().system('pip install msgpack==0.6.1')

get_ipython().system(' URL="https://github.com/openmined/PySyft.git" && FOLDER="PySyft" && if [ ! -d $FOLDER ]; then git clone -b dev --single-branch $URL; else (cd $FOLDER && git pull $URL && cd ..); fi;')

get_ipython().system('cd PySyft; python setup.py install  > /dev/null')

import os
import sys
module_path = os.path.abspath(os.path.join('./PySyft'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
get_ipython().system('pip install --upgrade --force-reinstall lz4')
get_ipython().system('pip install --upgrade --force-reinstall websocket')
get_ipython().system('pip install --upgrade --force-reinstall websockets')
get_ipython().system('pip install --upgrade --force-reinstall zstd')


# In[2]:


from __future__ import unicode_literals, print_function, division
from torch.utils.data import Dataset

import torch
from io import open
import glob
import os
import numpy as np
import unicodedata
import string
import random
import torch.nn as nn
import time
import math
import syft as sy
import pandas as pd
import random
from syft.frameworks.torch.federated import utils

from syft.workers import WebsocketClientWorker
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# In[3]:


get_ipython().system('wget https://download.pytorch.org/tutorial/data.zip  ')


# In[4]:


get_ipython().system('unzip data.zip')


# In[5]:


path = '/content/data/names/*.txt'

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)


#Load files in the path
def findFiles(path):
  return glob.glob(path)

#Read a file and then split to lines
def readLines(filename):
  lines =open(filename, encoding='utf-8').read().strip().split('\n')
  return [unicodeToAscii(line) for line in lines]

#Convert  string to ASCII format
def unicodeToAscii(s):
  return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
      and c in all_letters
  )



category_lines = {}
all_categories = []

for filename in findFiles(path):
  #print(filename)
  category = os.path.splitext(os.path.basename(filename))[0]
  all_categories.append(category)
  lines = readLines(filename)
  category_lines[category] = lines
  
n_categories = len(all_categories)

#for names in glob.glob(path):
  #print(names)
  
 
print("Number of categories: " + "\n" + str(n_categories))
print("\n" + "All categories: ")
print(*all_categories, sep = "\n")


# 

# In[6]:


print(*category_lines['Polish'][:6], sep = "\n")


# In[ ]:


class LanguageDataset(Dataset):
  # Constructor
  def __init__(self, text, labels, transform=None):
    self.data = text
    self.targets = labels # categories
    #self.to_torchtensor()
    self.transform = transform
    
  def to_torchtensor(self):
    self.data = torch.from_numpy(self.text, requires_grad=True)
    self.labels = torch.from_numpy(self.targets, requires_grad=True)
  
  # Returns length of dataset/batches
  def __len__(self):
    return len(self.data)
  
  # Returns data and target[torch tensor ]
  def __getitem__(self, idx):
    sample = self.data[idx]
    target = self.targets[idx]
    
    if self.transform:
      sample = self.transform(sample)
      
    return sample, target
    
  


# In[ ]:


# Arguments for the program
class Arguments():
  def __init__(self):
    self.batch_size = 1
    self.learning_rate = 0.005
    self.epochs = 10000
    self.federate_after_n_batches =15000
    self.seed = 1
    self.print_every = 200
    self.plot_every = 100
    self.use_cuda = False
    
args = Arguments()
    


# In[9]:


get_ipython().run_cell_magic('latex', '', '\n\\begin{split}\nnames\\_list = [d_1,...d_n]  \\\\\n\ncategory\\_list = [c_1,...c_n]\n\\end{split}')


# In[10]:


names_list = []
category_list = []

for nation, names in category_lines.items():
  for name in names:
    names_list.append(name)
    category_list.append(nation)
    
print(*names_list[:5], sep = "\n")
print(*category_list[:5], sep = "\n")
print("\n")
print("Data points loaded: " + str(len(names_list)))


# In[11]:


# An integer to every category
categories_numerical = pd.factorize(category_list)[0]

# Categories with tensor
category_tensor = torch.tensor(np.array(categories_numerical), dtype=torch.long)

categories_numpy = np.array(category_tensor)

print(names_list[100:120])
print(categories_numpy[100:120])


# We will turn every character in each input string into a vector, with a 1 marking that particular character present. <br>
# A word will just be a vector of character vectors and our RNN will process every character vector in the word.<br>
# This technique is called word embedding.

# In[ ]:


# This returns the index of a letter given
def letterToIndex(letter):
    return all_letters.find(letter)
    

# Turn a line into a <line_length x 1 x n_letters>
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor    
    
    
# Turn a list of strings into a list of tensors
def list_strings_to_list_tensors(names_list):
    lines_tensors = []
    for index, line in enumerate(names_list):
        lineTensor = lineToTensor(line)
        lineNumpy = lineTensor.numpy()
        lines_tensors.append(lineNumpy)
        
    return(lines_tensors)

lines_tensors = list_strings_to_list_tensors(names_list)


# In[13]:


# Testing the functions work
print(names_list[0])
print(lines_tensors[0])
print(lines_tensors[0].shape)


# In[ ]:


# Identify the longest word in the dataset as all tensors need to have the same
# shape 

max_line_size = max(len(x) for x in lines_tensors)

# Turn a line into a <line_length x 1 x n_letters>
def lineToTensorFillEmpty(line, max_line_size):
    tensor = torch.zeros(max_line_size, 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
        
    # If there is no character, a vector with (0,0,.... ,0) will be placed
    return tensor

# Turn a list of strings into a list of tensors using previous function
def list_strings_to_list_tensors_fill_empty(names_list):
    lines_tensors = []
    for index, line in enumerate(names_list):
        lineTensor = lineToTensorFillEmpty(line, max_line_size)
        lines_tensors.append(lineTensor)
    return(lines_tensors)

lines_tensors = list_strings_to_list_tensors_fill_empty(names_list)


# In[15]:


# Tensor shape check
print(lines_tensors[0].shape)


# In[16]:


# Create numpy array with all word embeddings
array_lines_tensors = np.stack(lines_tensors)
array_lines_proper_dimension = np.squeeze(array_lines_tensors, axis=2)

# Check array dimension
print(array_lines_proper_dimension.shape)


# In[17]:


def find_start_index_per_category(category_list):
    categories_start_index = {}
    
    #Initialize every category with an empty list
    for category in all_categories:
        categories_start_index[category] = []
    
    #Insert the start index of each category into the dictionary categories_start_index
    #Example: "Italian" --> 203
    #         "Spanish" --> 19776
    last_category = None
    i = 0
    for name in names_list:
        cur_category = category_list[i]
        if(cur_category != last_category):
            categories_start_index[cur_category] = i
            last_category = cur_category
        
        i = i + 1
        
    return(categories_start_index)

categories_start_index = find_start_index_per_category(category_list)

print(categories_start_index)


# In[ ]:


def randomChoice(l):
    rand_value = random.randint(0, len(l) - 1)
    return l[rand_value], rand_value


def randomTrainingIndex():
    category, rand_cat_index = randomChoice(all_categories) #cat = category, it's not a random animal
    #rand_line_index is a relative index for a data point within the random category rand_cat_index
    line, rand_line_index = randomChoice(category_lines[category])
    category_start_index = categories_start_index[category]
    absolute_index = category_start_index + rand_line_index
    return(absolute_index)


# In[19]:


#Two hidden layers, based on simple linear layers

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

#Let's instantiate the neural network already:
n_hidden = 128
#Instantiate RNN

device = torch.device("cuda" if args.use_cuda else "cpu")
model = RNN(n_letters, n_hidden, n_categories).to(device)
#The final softmax layer will produce a probability for each one of our 18 categories
print(model)


# In[ ]:


# Specify remote workers's location

hook = sy.TorchHook(torch)  # Hook PyTorch

# Uncomment this with the ip of each raspberry pi worker if you're using the
# raspberry pi and comment the block of code beneath this

# kwargs_websocket_alice = {"host": "ip_alice", "hook": hook}
# alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket_alice)
# kwargs_websocket_bob = {"host": "ip_bob", "hook": hook}
# bob = WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket_bob)

alice = sy.VirtualWorker(hook, id="alice")  
bob = sy.VirtualWorker(hook, id="bob")  

workers_virtual = [alice, bob]


# In[ ]:


# array_lines_proper_dimension = our data points(X)
# categories_numpy = our labels (Y)
langDataset = LanguageDataset(array_lines_proper_dimension, categories_numpy)

#assign the data points and the corresponding categories to workers.
federated_train_loader = sy.FederatedDataLoader(
    langDataset.federate(workers_virtual),
    batch_size=args.batch_size)


# # Model Training
# Now the data is processed, we'll start to train our RNN!

# In[ ]:


# Gives the category that corresponds to maximum predicted class probability
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

# Gives the amount of time passed since "since"
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Federated averaging
def fed_avg_every_n_iters(model_pointers, iter, federate_after_n_batches):
        models_local = {}
        
        if(iter % args.federate_after_n_batches == 0):
            for worker_name, model_pointer in model_pointers.items():
                # Assign model to the worker
                models_local[worker_name] = model_pointer.copy().get()
            model_avg = utils.federated_avg(models_local)
           
            for worker in workers_virtual:
                model_copied_avg = model_avg.copy()
                model_ptr = model_copied_avg.send(worker) 
                model_pointers[worker.id] = model_ptr
                
        return(model_pointers)     

def fw_bw_pass_model(model_pointers, line_single, category_single):
  
    # Get the right initialized model
    model_ptr = model_pointers[line_single.location.id]   
    line_reshaped = line_single.reshape(max_line_size, 1, len(all_letters))
    line_reshaped, category_single = line_reshaped.to(device), category_single.to(device)
    
    # Initialize hidden layer
    hidden_init = model_ptr.initHidden() 
    
    # And now zero the gradient
    model_ptr.zero_grad()
    hidden_ptr = hidden_init.send(line_single.location)
    amount_lines_non_zero = len(torch.nonzero(line_reshaped.copy().get()))
    
    # Forward passes
    for i in range(amount_lines_non_zero): 
        output, hidden_ptr = model_ptr(line_reshaped[i], hidden_ptr) 
    criterion = nn.NLLLoss()   
    loss = criterion(output, category_single) 
    loss.backward()
    
    model_got = model_ptr.get() 
    
    # Update model's weights 
    for param in model_got.parameters():
        param.data.add_(-args.learning_rate, param.grad.data)
        
        
    # Send the model
    model_sent = model_got.send(line_single.location.id)
    model_pointers[line_single.location.id] = model_sent
    
    return(model_pointers, loss, output)


# In[ ]:


# Training function
def train_RNN(n_iters, print_every, plot_every, federate_after_n_batches, list_federated_train_loader):
    current_loss = 0
    all_losses = []    
    
    model_pointers = {}
    
    # Send the initialized model to every single worker just before training
    for worker in workers_virtual:
        model_copied = model.copy()
        model_ptr = model_copied.send(worker) 
        model_pointers[worker.id] = model_ptr

    # Extract a random element from the list and perform training on it
    for iter in range(1, n_iters + 1):        
        random_index = randomTrainingIndex()
        line_single, category_single = list_federated_train_loader[random_index]
        line_name = names_list[random_index]
        model_pointers, loss, output = fw_bw_pass_model(model_pointers, line_single, category_single)
        
        # Update theloss
        loss_got = loss.get().item() 
        current_loss += loss_got
        
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
             
        # Print information on training
        # The name, guessed category, correct/incorrect and actual category
        if(iter % print_every == 0):
            output_got = output.get()
            guess, guess_i = categoryFromOutput(output_got)
            category = all_categories[category_single.copy().get().item()]
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, 
                                                    timeSince(start), 
                                                    loss_got, 
                                                    line_name, guess, correct))
            
            
    return(all_losses, model_pointers)


# ## Start the training

# In[24]:


# Turn the data points and categories into a list
list_federated_train_loader = list(federated_train_loader)

# Start the training
start = time.time()
all_losses, model_pointers = train_RNN(args.epochs, args.print_every, 
                                       args.plot_every, 
                                       args.federate_after_n_batches, 
                                       list_federated_train_loader)


# In[25]:


# Plot the loss we got during the training procedure
plt.figure()
plt.title("Loss over time training")
plt.ylabel("Loss")
plt.xlabel('Epochs (100s)')
plt.plot(all_losses)


# ## Predict!

# In[ ]:


def predict(model, input_line, all_categories, worker, n_predictions=3):
    """ 
    Uses :attr:`model` to predict top :attr:`n_predictions` categories 
    from :attr:`all_categories` for :attr:`input_line` using :attr:`worker`
  
    Parameters: 
        model (Module): model to be used for the prediction
        input_line (str): input to the model
        all_categories (list): list of all categories for the prediction
        worker(BaseWorker): worker where the prediction will be performed
        n_predictions(int): number of top predictions to return
  
    Returns: 
        list of tuples (value, category) sorted from max value to min
  
    """
    
    # copy the model to the worker only if is not already there
    if model.location.id != worker.id:
        model = model.copy().get()
        model_remote = model.send(worker)
    else:
        model_remote = model
    
    # convert the input_line to a tensor and send it to the worker
    line_tensor = lineToTensor(input_line)
    line_remote = line_tensor.copy().send(worker)

    # init the hidden layer
    hidden = model_remote.initHidden()
    hidden_remote = hidden.copy().send(worker)
        
    # get a result from the model
    with torch.no_grad():
        for i in range(line_remote.shape[0]):
            output, hidden_remote = model_remote(line_remote[i], hidden_remote)
    
    # get top N categories
    topv, topi = output.copy().get().topk(n_predictions, 1, True)

    # construct list of (value, category) tuples
    predictions = []
    for i in range(n_predictions):
        value = topv[0][i].item()
        category_index = topi[0][i].item()
        #print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])
        
    return predictions


# In[27]:


print(predict(model_pointers["alice"], "Qing", all_categories,  alice) )

