import numpy as np
import math

epsilon = 1e-9
l_squared = lambda target, prediction: np.sqrt(np.power(target-prediction, 2).sum())
dl_squared = lambda target, prediction: -(target-prediction)
#l_catlossentrop = lambda target, prediction: -(target*np.log(prediction + epsilon) + (1-target) * np.log(1-prediction + epsilon)).sum()
dl_catlossentrop = lambda target, prediction: np.nan_to_num(-target/(prediction + epsilon) + (1-target)/(1-prediction + epsilon), 0)
l_catlossentrop = lambda target, prediction: -(target*np.log(prediction+ epsilon)).sum()
#dl_catlossentrop = lambda target, prediction: -(target/(prediction + epsilon))
dl_catlossentropsoftmax = lambda target, prediction: prediction-target

#Used for Reinforcement Learning
l_maxlog = lambda reward, action: reward * action
dl_maxlog = lambda reward, action: (reward * np.log(action))

class AbstractLoss:
    def __init__(self, loss_function, dloss_function):
        self.__loss_function = loss_function
        self.__dloss_function = dloss_function
 
    def loss(self, target, prediction):
        return self.__loss_function(target, prediction)

    def dloss(self, target, prediction):
        return self.__dloss_function(target, prediction)

    def batch_loss(self, targets, predictions):
        return sum([self.loss(target, prediction) for target, prediction in zip(targets, predictions)])

    def batch_dloss(self, targets, predictions):
        return np.array([self.dloss(target, prediction) for target, prediction in zip(targets, predictions)])

class SSE(AbstractLoss):
    def __init__(self):
        super().__init__(l_squared, dl_squared)

class CategoricalCrossEntropy(AbstractLoss):
    def __init__(self):
        super().__init__(l_catlossentrop, dl_catlossentrop)

class LogError(AbstractLoss):
    def __init__(self):
        super().__init__(l_maxlog, dl_maxlog)

class CategoricalCrossEntropySoftmax(AbstractLoss):
    def __init__(self):
        super().__init__(l_catlossentrop, dl_catlossentropsoftmax)

    def batch_dloss(self, targets, predictions):
        return predictions - targets 

