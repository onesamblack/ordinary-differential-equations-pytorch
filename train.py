import os
import math
import uuid
import pandas as pd
import numpy as np
import torch
import pathlib
import pickle
import inspect
import wandb
import argparse
from dotenv import load_dotenv
from torch import nn 
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler




parser = argparse.ArgumentParser()
parser.add_argument("--smooth", default=True, help="apply smoothing (EWM=10)")
parser.add_argument("--scale", default=True, help="use minmax scaling on the timeseries")
parser.add_argument("--test_ratio", default=0.3, help="% of data to use as test, defaults to 30%")
parser.add_argument("--use_wandb", default=True, help="use wandb - defaults to True and don't change to false or I'll break")

args = parser.parse_args()


# api key - set in .env
load_dotenv()



USE_WANDB = args["use_wandb"]
PARAMETERS_ROOT_DIR = "/content"


data = pd.read_csv("daily_min_temp.csv")

smooth = data.iloc[:,1].ewm(span=10).mean().to_numpy()
orig = data.iloc[:, 1].to_numpy()



# scale
# this was originally in a notebook, hence keeping both scaler instances
scaler_smooth = MinMaxScaler()
scaler_orig = MinMaxScaler()



orig_scaled = scaler_orig.fit_transform(orig[:-2].reshape(-1, 1))
smooth_scaled = scaler_smooth.fit_transform(smooth[:-2].reshape(-1, 1))

test_size = args["test_ratio"]

# performing all the data preprocessing here makes sense in the context of a# notebook
train_data_original = orig[: math.ceil(len(orig) * (1 - test_size))]
train_data_original_scaled = orig_scaled[: math.ceil(len(orig) * (1 - test_size))]

test_data_original = orig[math.ceil(len(orig) * (1 - test_size)):]
test_data_original_scaled = orig_scaled[math.ceil(len(orig) * (1 - test_size)):]


train_data_smooth = smooth[: math.ceil(len(smooth) * (1 - test_size))]
train_data_smooth_scaled = smooth_scaled[: math.ceil(len(smooth) * (1 - test_size))]


test_data_smooth = smooth[math.ceil(len(smooth) * (1 - test_size)):]
test_data_smooth_scaled = smooth_scaled[math.ceil(len(smooth) * (1 - test_size)):]



assert train_data_original[1] != test_data_original[0]
print(f"train: {len(train_data_original)}, test: {len(test_data_original)}")


if args["smooth"]:
    use_for_training = "smooth"
else:
    use_for_training = "orig"

if args["scale"]:
    scaling = "scaled"
else: 
    scaling = "unscaled"


datasets ={
    "smooth": {
        "unscaled": (train_data_smooth, test_data_smooth),
        "scaled":  (train_data_smooth_scaled, test_data_smooth_scaled)
    },

    "orig": {
       "unscaled" : (train_data_original, test_data_original),
       "scaled":  (train_data_original_scaled, test_data_original_scaled),
    }
}

# generate the torch tensors for train and test
train = torch.FloatTensor(datasets[use_for_training][scaling][0])
test = torch.FloatTensor(datasets[use_for_training][scaling][1])

class DeviceTensor:
    """
    makes tensors based on the device (easier for switching between GPU and CPU 
    """
    def __init__(self, device):
        self.device = device
        self._tensors =  {
        "float": torch.cuda.FloatTensor if str(self.device) == "cuda" else torch.FloatTensor,
        "long": torch.cuda.LongTensor if str(self.device) == "cuda" else torch.LongTensor
        }

    def _make_tensor(self, tensor_type, *args, **kwargs):
        """
        returns a tensor appropriate for the device

        :param tensor_type: supported ['long', 'float']
        :returns: a torch.Tensor
        """
        t = self._tensors[tensor_type](*args,**kwargs).to(self.device)
        return t


class UnivariateWindowedTimeSeriesDataSet(Dataset, DeviceTensor):

  def __init__(self, X, window_size=1, device=None):
    super(UnivariateWindowedTimeSeriesDataSet, self).__init__(device)
    self.window_size = window_size
    self.X = X
    if device:
        self.device = device
    else:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  def __len__(self):
    return math.floor(len(self.X) / self.window_size)

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    if self.window_size > 1:
        _x = self._make_tensor("float", self.X[index:index+self.window_size])
        _y = self._make_tensor("float", self.X[(index +1) + self.window_size])
    else:
        _x = self._make_tensor("float", self.X[index])
        _y = self._make_tensor("float", self.X[index + 1])

    return _x, _y


class MultivariateWindowedTimeSeriesDataSet(Dataset, DeviceTensor):

  def __init__(self, X, window_size=1, device=None, autoencoder=False):
    super(UnivariateWindowedTimeSeriesDataSet, self).__init__(device)
    self.window_size = window_size
    self.X = X

    self.autoencoder=autoencoder
    if device:
        self.device = device
    else:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  def __len__(self):
    return math.floor(len(self.X) / self.window_size)

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    if self.window_size > 1:
        _x = self._make_tensor("float", self.X[index:index+self.window_size])
        if self.autoencoder:
            _y = self._make_tensor("float", self.X[index:index+self.window_size])
        else:
            _y = self._make_tensor("float", self.X[(index +1) + self.window_size])
    else:
        _x = self._make_tensor("float", self.X[index])
        if self.autoencoder:
            _y = self._make_tensor("float", self.X[index])
        else:
            _y = self._make_tensor("float", self.X[index + 1])
        

    return _x, _y



# some utility functions - this is organized terribly I know

def get_sampler(dataset: torch.utils.data.Dataset, shuffle: bool = True) -> torch.utils.data.Sampler:
    indices = list(range(0, len(dataset) -1))
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    return SubsetRandomSampler(indices)


def save_model(dir: str, 
			   model: nn.Module, 
			   epoch: int, 
			   name: str):

    """
    Saves model parameters
    
    Parameters
    ----------
    dir : str
        directory. saves each set of parameters under model_name/parameters/epoch.pt
    model : nn.Module
        a pytorch module
    epoch : int
        epoch number
    name : str
        the name of the model - overwrite this if not using wandb
    """
    pathlib.Path(f"{dir}/{name}/parameters/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{dir}/{name}/model/").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{dir}/{name}/parameters/_{epoch}.pt")
    with open(f"{dir}/{name}/model/args.pkl", "wb") as pickle_file:
        pickle.dump(model.arg_dict(), pickle_file)
    
    

    print(f"saved {name} for epoch: {epoch}", flush=True)



def train_it(epoch, device):

    model.train()
    train_loss = 0
    for i, data in enumerate(train_loader):
        x1s, labels = data[0], data[1]
        preds = model(x1s)
        
        optimizer.zero_grad()
        loss = loss_fn(preds, labels)
        loss.backward()
        

        optimizer.step()

        train_loss += loss.item()

    
    per_item_loss = train_loss / len(train_loader)

    print(f"train_loss_per_item: {per_item_loss}")
    print(f"train_loss: {train_loss}")
    
    if USE_WANDB:
	    wandb.log({"train_loss_per_item": per_item_loss})
	    wandb.log({"train_loss": train_loss})
    return per_item_loss, train_loss 

def validate_it(epoch,device):

    print(f"running validation epoch: {epoch}", flush=True)
    test_loss = 0
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x1s,labels = data[0], data[1]
            test_preds = model(x1s)
            test_loss += loss_fn(test_preds, labels).item()
    per_item_loss = test_loss / len(test_loader)
    print(f"test_loss_per_item: {per_item_loss}")
    print(f"test_loss: {test_loss}")

    if USE_WANDB:
	    wandb.log({"test_loss_per_item": per_item_loss})
	    wandb.log({"test_loss": test_loss})
     
    return per_item_loss, test_loss
 




# some models

class LSTM(nn.Module, DeviceTensor):
  """implements an lstm - a single/multilayer uni/bi directional lstm"""
  def __init__(self, n_features, window_size, 
               output_size, h_size, n_layers=1, 
               bidirectional=False, dropout=0, device=torch.device('cpu'), initializers=[]):
    super(LSTM, self).__init__()
    DeviceTensor.__init__(self, device)
    self.n_features = n_features
    self.window_size = window_size
    self.output_size = output_size
    self.h_size = h_size
    self.n_layers = n_layers
    self.directions = 2 if bidirectional else 1
    self.device = device
    self.model_name = "LSTM"
    self.dropout= dropout
    # save input args to recreate model for predictions
    self._args = dict(
        n_features = n_features,
        window_size = window_size,
        output_size = output_size,
        h_size = h_size,
        n_layers = n_layers,
        bidirectional = bidirectional,
        dropout=dropout
    )

    self.lstm = nn.LSTM(input_size=n_features, hidden_size=h_size, 
                        num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
    self.hidden = None
    
    layer_dim_1 = int(self.h_size / 2 if self.h_size > 32 else 32)
    layer_dim_2 = int(layer_dim_1 / 2)

    self.linear = nn.Linear(self.h_size * self.directions, layer_dim_1)
    self.linear2 = nn.Linear(layer_dim_1, layer_dim_2)
    self.linear3 = nn.Linear(layer_dim_2, output_size)
    self.linear_activation = nn.ReLU()

    self.layers =  [self.lstm, self.linear, self.linear2, self.linear3]
    self.initializers = initializers
    
    self._initialize_all_layers()

  
  def arg_dict(self):
      return self._args

  
  def _initialize_all_layers(self):
    """
    
    If the user provides initializers, initializes each layer[i] with initializer[i]
    If no initializers provided, moves the layers to the device specified
    """
    if all([self.initializers, len(self.initializers) != self.layers, 
            len(self.initializers) == 1]):  
      warnings.warn("only one initializer: {} was provided for {} layers, the initializer will be used for all layers"\
                    .format(len(self.layers)))
      if len(self.initializers) == 1:
        [self._initialize_layer(self.initializers[0],x) for x in self.layers]
      else:
        [self._initialize_layer(self.initializers[i],x) for i, x in enumerate(self.layers)]
    
    elif all([self.initializers, 
              len(self.initializers) != self.layers]):
      raise Exception("{} initializers were provided for {} layers, need to provide an initializer for each layer"\
                      .format(len(self.initializers), len(self.layers)))
    else:
      # uses default initialization, but moves layer to device
      [self._initialize_layer(None, x) for x in self.layers]

  def _initialize_layer(self, initializer, layer):
    if initializer:
      pass
      #todo - add some dynamic initialization methods
    layer.to(self.device)


  def init_hidden(self, batch_size):
    
    hidden_a  = torch.randn(self.n_layers * self.directions,
                            batch_size ,self.h_size).to(self.device)
    hidden_b = torch.randn(self.n_layers * self.directions, 
                           batch_size,self.h_size).to(self.device)
    
    hidden_a = Variable(hidden_a)
    hidden_b = Variable(hidden_b)

    return (hidden_a, hidden_b) 

  def forward(self, input):
    batch_size =  list(input.size())[0]
    self.hidden = self.init_hidden(batch_size)
    
    lstm_output, self.hidden = self.lstm(input, self.hidden) 
    last_hidden_states = torch.index_select(lstm_output, 1,  index=self._make_tensor("long", ([self.window_size-1])))
    
    x = self.linear_activation(self.linear(last_hidden_states))
    x = self.linear_activation(self.linear2(x))
    return self.linear3(x)  

  def get_py_source(self):
      return inspect.getsource(self.__class__)


class NeuralODEForecaster(nn.Module, ):
    """
    the NeuralODEForecaster is hiding
    """
    def __init__(self):
        super(NeuralODEForecaster,self).__init__()


    def forward(self, x, t):
        pass





# Training


USE_WANDB = True
PARAMETERS_ROOT_DIR = "/content"


sweep_config = {
  "name": "lstm_sweep",
  "method": "grid",
  "parameters": {
        "lr": {
            "values": [1e-6, 1e-5, 1e-4, 1e-3]
        },
        "batch_size": {
            "values": [8, 16, 32, 64]
        },
        "window_size": {
            "values": range(0,20,2)
        },
        "hidden_size": {
            "values": range(32,512,32)
        }
    }
}


config = dict(
    lr = 1e-3,
    batch_size = 1,
    layers = 2, 
    window_size = 10,
    hidden_size = 128,
    dropout = .5,
    epochs = 50,
    smoothing_span = 20

)


if USE_WANDB:
    wandb.sdk.login(key=os.environ["WANDBKEY"])
    wandb.init(project="neural_ode_temperature", config=config)



train_data = UnivariateWindowedTimeSeriesDataSet(train, 
                                                 config["window_size"])
test_data = UnivariateWindowedTimeSeriesDataSet(test, 
                                                config["window_size"])

train_loader = DataLoader(train_data, batch_size=config["batch_size"], 
                          sampler=get_sampler(train_data))
test_loader = DataLoader(test_data, batch_size=config["batch_size"], 
                         sampler=get_sampler(test_data))


_DEVICE = torch.device(f"cuda:{0}")
_PARAMETERS_ROOT_DIR = "/content"


model = LSTM(n_features= 1, 
             window_size = config["window_size"],
             n_layers = config["layers"], 
             output_size=1,
             h_size=config["hidden_size"],
             dropout=config["dropout"],
             device = _DEVICE)


optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])  
loss_fn = nn.MSELoss()

model.to(_DEVICE)


min_test_loss = [math.inf, None]



for epoch in range(1, config["epochs"]+1):
    try:
        train_it(epoch, _DEVICE)
        l1, l2 = validate_it(epoch, _DEVICE)
        if l1 < min_test_loss[0]:
            min_test_loss[0] = l1
            min_test_loss[1] = epoch
        save_model(f"{PARAMETERS_ROOT_DIR}/runs", model, epoch,\
                          name= wandb.run.name if USE_WANDB else str(uuid.uuid4()))
    except KeyboardInterrupt:
        print(f"{wandb.run.name}, {min_test_loss[1]}")
        raise
    except:
        raise
print(f"\"{wandb.run.name}\", {min_test_loss[1]}")

