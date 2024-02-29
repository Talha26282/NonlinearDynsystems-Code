import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import optimizers
from keras.callbacks import History
from scipy.io import loadmat
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Lambda
from keras import backend as K
from keras.constraints import Constraint
import random

print("Num GPUs Available: ", len(tf.test.gpu_device_name()))
    
timestep=0.01 #not used
dim=50 #No. of dimensions considered for the input and output data
modes=30 #No. of eigenvalues to be found (always even)
conjpairnum=int(modes/2) #Number of conjugate pairs assumed
  
#loading data from matlab
X_train=loadmat('mach2proj4data_111023.mat')['X1']
y_train=loadmat('mach2proj4data_111023.mat')['X2']

D_dmd=loadmat('Dmd_initvalsmach2.mat')['D1neu']
V_dmd=loadmat('Dmd_initvalsmach2.mat')['V1neu']
W_dmd=loadmat('Dmd_initvalsmach2.mat')['W1neu1']

X_train=np.transpose(X_train)
y_train=np.transpose(y_train)
D_dmd1=np.transpose(D_dmd)

# Include good weight initialization here
initializer = tf.keras.initializers.GlorotNormal(seed=None)

history1=History()
temp=0
tempstat=[]
#Designing a custom neural network (taking complex values into account) 
currstat1= tf.keras.Input(shape=(dim,),dtype=tf.float64) #current state
while temp<conjpairnum:
    s_nodelay=tf.keras.layers.Dense(2,activation='linear',use_bias=False,dtype=tf.float64)(currstat1)
    tempstatlay=tf.keras.layers.Dense(2,activation='linear',use_bias=False,dtype=tf.float64)(s_nodelay)
    tempstat.append(tempstatlay) 
    temp=temp+1
tempstatconc=tf.keras.layers.Concatenate()(tempstat)
nextstat=tf.keras.layers.Dense(dim,activation='linear',use_bias=False,dtype=tf.float64)(tempstatconc)
model = tf.keras.Model(inputs=currstat1, outputs=nextstat)
print(model.summary())

#Initializing using DMD weights
#For W's
# D_dmd1[5,5]=-1*D_dmd1[5,5]
for m in range(conjpairnum):
    l=[] #For the weights for s1
    x=W_dmd[0+2*m:2+2*m]
    x=np.transpose(x)
    l.append(x)
    model.layers[m+1].set_weights(l) #loaded_model.layer[0] being the layer

#For D's
for m in range(conjpairnum):
    l1=[] #For the weights for s1
    x1=D_dmd1[0+2*m:2+2*m,0+2*m:2+2*m]
    # x=np.transpose(x)
    l1.append(x1)
    model.layers[conjpairnum+m+1].set_weights(l1) #loaded_model.layer[0] being the layer


#For V's
l2=[] #For the weights for s1
x2=V_dmd
x2=np.transpose(x2)
l2.append(x2)
model.layers[2*conjpairnum+2].set_weights(l2) #loaded_model.layer[0] being the layer

#Need to modify everything to be scalable especially the weight matrices generated
#To impose orthogonality on left and right eigen vectors, let us build a custom loss function


class CustomMSE(keras.losses.Loss):
    def __init__(self, Weights, name="custom_mse"):
        super().__init__(name=name)
        self.w = Weights
       
    def call(self, y_true, y_pred):        
        Weights=model.weights
        V=tf.transpose(Weights[2*conjpairnum])
        listtemp=[]
        for lay in range(conjpairnum):
            listtemp.append(Weights[lay])
        W=tf.transpose(tf.concat(listtemp,1))
        #First involving mse loss term  
        mean_squared_difference = tf.keras.metrics.mean_squared_error(y_true, y_pred)
        
        #Computing the orthogonality loss
        #First need to compute the orth_matrix
        I1=np.eye(modes)
        # I1=np.eye(modes)*0.5
        # for i in range(modes):
        #     for j in range(modes):
        #         if i==j and np.mod(j,2)!=0:
        #             I1[i,j]=-I1[i,j]             
        I1=tf.convert_to_tensor(I1,dtype=tf.float64)         
        orth_loss=tf.norm(tf.matmul(W, V)-I1)
        
        #Computing the high frequency eigenvalue penalty
        penal2sum=0
        penal2sum1=0
        for lay in range(conjpairnum):
            eigvals=tf.linalg.eig(Weights[conjpairnum+lay])
            Eig=eigvals[0][0]
            Eig1=eigvals[0][1]
            
            if lay>-1 and lay<5:
                compeig2=tf.abs(tf.math.atan2(tf.math.imag(Eig),tf.math.real(Eig)))
                comp3=tf.abs(compeig2-0.5519)
                compeig3=tf.abs(tf.math.atan2(tf.math.imag(Eig1),tf.math.real(Eig1)))
                comp4=tf.abs(compeig3-0.5519)
                penal2sum1=penal2sum1+comp3+comp4
            
            compeig=tf.abs(Eig)
            comp1=tf.abs(compeig-1)
            compeig1=tf.abs(Eig1)
            comp2=tf.abs(compeig1-1)
            penal2sum=penal2sum+(comp1+comp2)
            # else:
            #     # zerotf=tf.convert_to_tensor(I1,dtype=tf.float64) 
            #     # comp1=tf.math.atan2(zerotf,Eig1)
                
            #     comp=tf.abs(tf.math.atan2(tf.math.imag(Eig),tf.math.real(Eig)))
            #     comp1=tf.abs(tf.math.atan2(tf.math.imag(Eig1),tf.math.real(Eig1)))
            #     penal2sum=penal2sum+1e-6*(comp+comp1)
      
        #tf.print(penal2sum)
        #tf.print(orth_loss)
        # tot_loss=12*mean_squared_difference+orth_loss+1e-2*penal2sum+2e-1*penal2sum1;
        tot_loss=10*mean_squared_difference+orth_loss+1e-2*penal2sum+2e-1*penal2sum1; #12
        return tot_loss
    
mbgd = optimizers.Adam(learning_rate=2e-2) #0.001

#For checking the network output (ensure it is working correctly)
outputscheck = [K.function([model.input], [layer.output])([X_train]) for layer in model.layers]

Weights=model.get_weights()
listtemp=[]
for lay in range(conjpairnum):
    listtemp.append(Weights[lay])
V1=np.transpose(np.asmatrix(np.reshape([Weights[2*conjpairnum]],[modes,dim])))
W1=np.transpose(np.asmatrix(np.reshape(np.concatenate(listtemp,axis=1),[dim,modes])))

#First need to compute the orth_matrix
I1=np.eye(modes)
orth_check=np.linalg.norm(W1*V1-I1)

#Training
history = History()
print('Training:')
epochnum=75 #180
for i in range(epochnum):
    #network specs for Training  
    Weights=model.weights
    
    model.compile(optimizer=mbgd,loss=CustomMSE(Weights))    
    model.fit(x=X_train,y=y_train,batch_size=128,epochs=1,callbacks=[history,history1]) #128
print('Training done')
    
Weights=model.weights
V=tf.transpose(Weights[2*conjpairnum])
listtemp=[]
for lay in range(conjpairnum):
    listtemp.append(Weights[lay])
W=tf.transpose(tf.concat(listtemp,1))

#Computing the orthogonality loss
#First need to compute the orth_matrix
I1=np.eye(modes)   
I1=tf.convert_to_tensor(I1,dtype=tf.float64)         
orth_loss=tf.norm(tf.matmul(W, V)-I1)

tf.print(orth_loss)

checkorth=np.asmatrix(W_dmd)*np.asmatrix(V_dmd)
checkorth1=W1*V1

Weights=model.get_weights()
listtemp=[]
for lay in range(conjpairnum):
    listtemp.append(Weights[lay])
V1=np.transpose(np.asmatrix(np.reshape([Weights[2*conjpairnum]],[modes,dim])))
W1=np.transpose(np.asmatrix(np.reshape(np.concatenate(listtemp,axis=1),[dim,modes])))

#First need to compute the orth_matrix
I1=np.eye(modes)
orth_check1=np.linalg.norm(W1*V1-I1)

#For checking the network output (ensure it is working correctly)
outputscheck1 = [K.function([model.input], [layer.output])([X_train]) for layer in model.layers]

M1=history1.history
#Checking training
plt.plot(M1['loss'])
plt.xlabel('number of iterations')
plt.ylabel('mse loss')
# plt.title('Training loss for Simple model (1st order of accuracy)')
plt.show()

#Finding the system matrix through weights learned
Dall=np.zeros([modes,modes])
Wall=Weights[0]
for k in range(conjpairnum):
    if k<conjpairnum-1:
        Wall=np.concatenate((Wall,Weights[k+1]),axis=-1)
    Dall[2*k:2*k+2,2*k:2*k+2]=Weights[conjpairnum+k]

sysmatneu=np.transpose(np.asmatrix(Weights[modes]))*np.transpose(np.asmatrix(Dall))*np.transpose(np.asmatrix(Wall))
matcheck=sysmatneu*np.transpose(np.asmatrix(X_train))

np.savetxt('Aneutcdynmach2.txt',sysmatneu, delimiter=' ')

checkeig=np.linalg.eig(Dall)
