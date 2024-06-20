import os
import numpy as np
import cv2

from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense
from keras.models import Model

is_init = False
size = -1
X=[]
y=[]
label = [] # or hme jab label ki jaruart pdegi un words ki to hm dorety isse kam chalengye
dictionary = {} # asscoaited at unique int with each word
c = 0  # var jo unique vlaue hm aaisgn krenge har ek word ko

for i in os.listdir():  # will give all the file in  directory we demand
    if i.split(".")[-1] == "npy" and not (i.split(".")[0] == "labels"):  # npy se jo phel likha h use split krke file name dkeh ajiske last me npy ho uni se kam h or ba d vala kyu label ke last me bhi npy h use read thodi krna h bhai vo kam ki nhi h to skip krne ke liye bas
    # if i.split(".")[-1] == "npy":  # npy se jo phel likha h use split krke file name dkeh ajiske last me npy ho uni se kam h
        if not (is_init):
            is_init = True
            X = np.load(i)   # loading each file in a  in  list
            size = X.shape[0]   # 0 isliye number of rows we have
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1) # ye bs ek var ki  list thiise sixe se into kr ke size ki size ban diya
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1))) # [3000,] esa aa rh ajb y ki size ko print kri to

#              # ab yha akr ek or dikkat y ki jo vlaue h vo string h modle string pe wirk nhi krega to hme hr string ko wk unique alue aasign krni pdegi label vo ham dictnory ki help se kreng eusme hme label lsit ki bhi jarurat odegi

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c = c + 1
# print(X)
# print(y)
print(dictionary)
# now we have to convert this y to integer isted of str
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32") # data tye str hi print kr rh a to data type change kra
# print(y.shape)
# print(y)
###  hello = 0 nope = 1 ---> [1,0] ... [0,1]
# in int 1 is greater than 0  but nope in not greater than hello in any way you cant comare two diffrent menaigns but you can't compare or isse hmare modle hello ko kam president dega or nope ko jyda we don't want to do that
# to solve this when we are ointing to 0 we will create 2 d array when the idex is 0 the value is 0 will say 1,0 and where 1 is there will turn first index to one 0,1
#  how to do that very esay - fun-tensorflow.keras.utils import to_categorical

y = to_categorical(y)
# print(y.shape) # phle y ki vlaue single col thi ab hmne use convert krke 7 col krdi
# print(y)

#  one thing you willbe notie that first we train model on  hello than noe mtlb y me jo data h vo continously store ham esa ni chahte to hme data ko uer niceh mix krna hoga to kro fir
#  so modle can predict very effectively

X_new = X.copy()
y_new = y.copy()
counter = 0
#
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt) # it will going to suffle all this values
#
for i in cnt:
    X_new[counter] = X[i] # counter postion mtlb 0th postion  or usko i se intaialize kr diya particualr suffle value  esa kyu kiya we dont want to lose our data
    y_new[counter] = y[i]
    counter = counter + 1
# print(y)
# print(y_new)
ip = Input(shape=(X.shape[1],))
# #
m = Dense(512, activation="relu")(ip) # inka meanign chat gpt krna
m = Dense(256, activation="relu")(m)
#
op = Dense(y.shape[1], activation="softmax")(m) # why ase shape ke ander 1 in y output we want 3 nerons meaning uh output neuron which is making prediction with 0th idex first index and lsat index and third index the first winning  the numbers of columsns wh have the activation
 # why softmax is used so we want every neuron having the probability associated with beign first class,seond class  when all of this summed up one that we are interesting to do and we want to connect it woth revious middle layer we have
model = Model(inputs=ip, outputs=op)
#
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc']) # why "categorical_crossentropy becoz we have categorial data
#
model.fit(X, y, epochs=50)

model.save("model.h5")
np.save("labels.npy", np.array(label))