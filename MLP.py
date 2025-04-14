# -*- coding: utf-8 -*-

"""
@ author: Tiancheng Xing
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP(object):
    def __init__(self,_in,_out1,_out2,outSize):
        self.W1 = np.random.randn(_in,_out1)
        self.b1 = np.zeros((1,_out1))        

        self.W2 = np.random.randn(_out1,_out2) 
        self.b2 = np.zeros((1,_out2))        

        self.W3 = np.random.randn(_out2,outSize) 
        self.b3 = np.zeros((1,outSize))         
        
    def forward(self,X):
        self.Z1 = np.dot(X,self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)

        self.Z2 = np.dot(self.A1,self.W2) + self.b2 
        self.A2 = sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2,self.W3) + self.b3 
        self.A3 = sigmoid(self.Z3)

        return self.A3 

    def backpropagate(self,X,Y,learning_rate=0.1):
        dA3 = self.A3 - Y 
        dZ3 = dA3 * sigmoid_derivative(self.A3)

        dW3 = np.dot(self.A2.T,dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = np.dot(dZ3,self.W3.T) 
        dZ2 = dA2 * sigmoid_derivative(self.A2) 

        dW2 = np.dot(self.A1.T,dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2,self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.A1)

        dW1 = np.dot(X.T,dZ1) 
        db1 = np.sum(dZ1,axis=0,keepdims=True) 

        self.W1 -= learning_rate * dW1 
        self.b1 -= learning_rate * db1 
        self.W2 -= learning_rate * dW2  
        self.b2 -= learning_rate * db2  
        self.W3 -= learning_rate * dW3  
        self.b3 -= learning_rate * db3  


if __name__ == "__main__":
    digits = load_digits()
    X = digits.data / 16.0
    Y = digits.target.reshape(-1,1)

    encoder = OneHotEncoder(sparse_output=False)

    Y = encoder.fit_transform(Y)


    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    _in = X.shape[1]
    _out1 = 64
    _out2 = 32
    outSize = 10

    nn = MLP(_in,_out1,_out2,outSize)

    epochs = 500
    learning_rate = 0.0001

    loss_history = []

    for epoch in range(epochs):
        out = nn.forward(X_train)
        loss = np.mean(np.square(Y_train - out))
        loss_history.append(loss)
        nn.backpropagate(X_train,Y_train,learning_rate)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    y_pred = nn.forward(X_test)
    acc = np.mean(np.argmax(y_pred,axis=1) == np.argmax(Y_test,axis=1))
    print(f"\nAccuracy on Test Set: {acc*100:.2f}%")
