# -*- coding: utf-8 -*-
"""
@ author: Tiancheng Xing
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

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

        self.mW1,self.vW1 = np.zeros_like(self.W1),np.zeros_like(self.W1)
        self.mb1,self.vb1 = np.zeros_like(self.b1),np.zeros_like(self.b1)
        self.mW2,self.vW2 = np.zeros_like(self.W2),np.zeros_like(self.W2)
        self.mb2,self.vb2 = np.zeros_like(self.b2),np.zeros_like(self.b2)
        self.mW3,self.vW3 = np.zeros_like(self.W3),np.zeros_like(self.W3)
        self.mb3,self.vb3 = np.zeros_like(self.b3),np.zeros_like(self.b3)
        self.t = 0     

    def __adam_update(self, param, dparam, m, v, beta1, beta2, lr, epsilon=1e-8):
        self.t += 1
        m[:] = beta1 * m + (1 - beta1) * dparam
        v[:] = beta2 * v + (1 - beta2) * (dparam ** 2)

        m_hat = m / (1 - beta1 ** self.t)
        v_hat = v / (1 - beta2 ** self.t)

        return param - lr * m_hat / (np.sqrt(v_hat) + epsilon)

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

        beta1,beta2 = 0.9,0.999
        self.W1 = self.__adam_update(self.W1,dW1,self.mW1,self.vW1,beta1,beta2,learning_rate)
        self.b1 = self.__adam_update(self.b1,db1,self.mb1,self.vb1,beta1,beta2,learning_rate)
        self.W2 = self.__adam_update(self.W2,dW2,self.mW2,self.vW2,beta1,beta2,learning_rate)
        self.b2 = self.__adam_update(self.b2,db2,self.mb2,self.vb2,beta1,beta2,learning_rate)
        self.W3 = self.__adam_update(self.W3,dW3,self.mW3,self.vW3,beta1,beta2,learning_rate)
        self.b3 = self.__adam_update(self.b3,db3,self.mb3,self.vb3,beta1,beta2,learning_rate)

    def save_parameters(self,path):
        np.savez(path,W1=self.W1,b1=self.b1,W2=self.W2,b2=self.b2,W3=self.W3,b3=self.b3)

    def load_parameters(self,path):
        data = np.load(path)
        self.W1,self.b1 = data["W1"],data["b1"]
        self.W2,self.b2 = data["W2"],data["b2"]
        self.W3,self.b3 = data["W3"],data["b3"]

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

    epochs = 1000
    learning_rate = 0.001
    early_stop_patience = 50
    model_path = "mlp_model.npz"

    loss_history = []
    acc_history = []
    best_loss = float("inf")
    stop_counter = 0

    for epoch in range(epochs+1):
        out = nn.forward(X_train)
        loss = np.mean(np.square(Y_train - out))
        loss_history.append(loss)

        y_train_pred = np.argmax(out,axis=1)
        y_train_true = np.argmax(Y_train,axis=1)
        acc = accuracy_score(y_train_true,y_train_pred)
        acc_history.append(acc)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss = {loss:.4f} | Accuracy = {acc*100:.2f}%")

        nn.backpropagate(X_train,Y_train,learning_rate)

        if loss < best_loss:
            best_loss = loss
            nn.save_parameters(model_path)
            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    nn.load_parameters(model_path)
    y_pred = nn.forward(X_test)
    y_test_label = np.argmax(Y_test,axis=1)
    y_pred_label = np.argmax(y_pred,axis=1)

    acc = accuracy_score(y_test_label,y_pred_label)
    print(f"\nAccuracy on Test Set: {acc*100:.2f}%\n")
    print("Classification Report:\n")
    print(classification_report(y_test_label,y_pred_label))

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(loss_history,label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(acc_history,label="Train Accuracy",color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    cm = confusion_matrix(y_test_label,y_pred_label)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
    plt.title("Confusion Matrix on Test Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
