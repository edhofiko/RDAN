import numpy as np
import copy
import pickle
import time
class AbstractModel:
    def __init__(self):
        pass
 
    def forward(self, X):
        pass

    def backward(self, X, derr):
        pass
   
    def copy(self):
        return copy.deepcopy(self)

    def save(self, filedir):
        pickle.dump(self, open(filedir, "wb"))

    def load(filedir):
        return pickle.load(open(filedir, "wb"))
    
    def epoch_end(self):
        pass

    def fit(self, X, Y, batch, verbose):
        pass


class Sequential(AbstractModel):
    def __init__(self, layers):
        self.__layers = layers

    def forward(self, X):
        Z = X
        for layer in self.__layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, X, derr):
        Zs = []
        Z = X
        for layer in self.__layers:
            Zs.append(Z)
            Z = layer.forward(Z)
        for i in range(len(self.__layers)-1, -1, -1):
            derr = self.__layers[i].backward(Zs[i], derr)
        return derr
        
    def batch_forward(self, X):
        Z = X
        for layer in self.__layers:
            Z = layer.batch_forward(Z)
        return Z 

    def batch_backward(self, X, derr):
        Zs = []
        Z = X
        for layer in self.__layers:
            Zs.append(Z)
            Z = layer.batch_forward(Z)
        for i in range(len(self.__layers)-1,-1,-1):
            derr = self.__layers[i].batch_backward(Zs[i], derr)
        return derr

    def batch_train(self, X, Y, loss):
        Zs = []
        Z = X
        for layer in self.__layers:
            Zs.append(Z)
            Z = layer.batch_forward(Z)
        derr = loss.batch_dloss(Y, Z)
        l = loss.batch_loss(Y, Z)
        for i in range(len(self.__layers)-1,-1,-1):
            derr = self.__layers[i].batch_backward(Zs[i], derr)
        return derr, l

    def epoch_end(self):
        for layer in self.__layers:
           layer.epoch_end()

    def fit(self, X, Y, Loss, batch=32, max_epoch=100, min_loss=1e-2, verbose=True):
        epoch = 0
        loss = float("inf")
        
        while(epoch < max_epoch and loss > min_loss):
           loss = 0
           train_data = [*zip(X, Y)]
           np.random.shuffle(train_data)
           train_data = np.array(train_data[:-(len(train_data) % batch) or None]).reshape(int(len(train_data)/batch), batch, 2).tolist() + [train_data[-(len(train_data) % batch):]]
           count = 0

           for index, data in enumerate(train_data):
              x, y = zip(*data)
              x, y = np.array(x), np.array(y)
              z = self.batch_forward(x)
              derr = Loss.batch_dloss(y, z)
              self.batch_backward(x, derr)
              l = Loss.batch_loss(y, z)
              loss += l
              count += len(data)
              print(epoch, str(index+1)+"/"+str(len(train_data)), loss/(count), l/len(data), end='\r')
              self.epoch_end()
           epoch+=1
           print()


class SingletonSequential(Sequential):
    def __init__(self, layers):
        self.__layers = layers
        self.__Z = [None] * (len(layers)+1)

    def forward(self, X):
        Z = X
        for layer in self.__layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, X, derr):
        Zs = []
        Z = X
        for layer in self.__layers:
            Zs.append(Z)
            Z = layer.forward(Z)
        for i in range(len(self.__layers)-1, -1, -1):
            derr = self.__layers[i].backward(Zs[i], derr)
        return derr
        
    def batch_forward(self, X):
        Z = X
        for layer in self.__layers:
            Z = layer.batch_forward(Z)
        return Z 

    def batch_backward(self, X, derr):
        Zs = []
        Z = X
        for layer in self.__layers:
            Zs.append(Z)
            Z = layer.forward(Z)
        for i in range(len(self.__layers)-1,-1,-1):
            derr = self.__layers[i].batch_backward(Zs[i], derr)
        del Z
        return derr

    def singleton_train(self, X, Y, loss):
        Z = X
        self.__Z[0] = Z
        for i, layer in enumerate(self.__layers):
           Z = self.__Z[i+1] = layer.batch_forward(Z, out=self.__Z[i+1])

        derr = loss.batch_dloss(Y, Z)
        l = loss.batch_loss(Y, Z)
        
        for i in range(len(self.__layers)-1,-1,-1):
            derr = self.__layers[i].batch_backward(self.__Z[i], derr, out=self.__Z[i])


        return derr, l
        
    def singleton_fit(self, X, Y, Loss, batch=32, max_epoch=100, min_loss=1e-2, verbose=True):
        epoch = 0
        loss = float("inf")
        
        while(epoch < max_epoch and loss > min_loss):
           loss = 0
           train_data = [*zip(X, Y)]
           np.random.shuffle(train_data)
           train_data = np.array(train_data[:-(len(train_data) % batch) or None]).reshape(int(len(train_data)/batch), batch, 2).tolist() + [train_data[-(len(train_data) % batch):]]
           count = 0

           for index, data in enumerate(train_data):
              x, y = zip(*data)
              x, y = np.array(x), np.array(y)
              _, l = self.singleton_train(x, y, Loss)
              loss += l
              count += len(data)
              print(epoch, str(index+1)+"/"+str(len(train_data)), loss/(count), l/len(data), end='\r')
           epoch+=1
           print()
        
        
