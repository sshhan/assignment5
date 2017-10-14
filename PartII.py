#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from FisherFace import*
import matplotlib.pyplot as plt
import cv2
########################################
def data_new(k,W,A,m):
    # Function newA=data_new(k,W,A)
    # compute the reducing matrix
    # Wk: the matrix composed of the first k eigenvectors
    # A: is the list of image vectors- mean of column of A
    b1=A.shape[1]
    A=A - np.tile(m, (b1,1)).T
    Wk=W[:,:k]
    newA=np.dot(Wk.T,A)
    return newA
#########################################
def centerPCA(A):
    centrA=np.zeros((10,30))
    b1=A.shape[1]
    AT=A.T
    for i in range(b1):
        ii=i//12
        centrA[ii]=centrA[ii]+AT[i]
    return centrA.T/12
######################################
def confusionmatrix(A,B):
    cm=np.zeros((10,10)) 
    b1=B.shape[1]
    distance=np.zeros(10)
    AT=A.T
    BT=B.T
    for i in range(b1):
        for j in range(10): #test 
            distance[j]=np.linalg.norm(AT[j] - BT[i])  #distance#print(dis)
        order_index = np.argsort(distance) #sort
        index = order_index[0]  # find the index of the min
        cm[ i//12 ][ index ] = cm[ i//12 ][ index ] + 1
    return cm
#################################
def fusiondata(A,B,alpha):
    C=np.vstack((alpha*A,(1-alpha)*B))
    return C
##########################
'''1'''
#####oringal data###
trdata,trlabel=read_faces('train')
tedata,telabel=read_faces('test')

####PCA data######
Wtr,Ltr,mtr=myPCA(trdata)
tr=data_new(30,Wtr,trdata,mtr)
te=data_new(30,Wtr,tedata,mtr)
#the class-mean of PCA is: Z  ##
Z=centerPCA(tr,trlabel,30)
PCAcm=confusionmatrix(Z,te)
AR_pca=np.trace(PCAcm)/sum(sum(PCAcm))
print('the cofusion matrix of PCA is:\n',PCAcm)
print('the accurate of PCA is:',round(AR_pca,3))
###########eigenvalues-faces################
'''2'''
#mean-face
fig=plt.figure()
plt.gray()
plt.subplot(3,3,9)
mtr1=float2uint8(mtr)
plt.imshow(mtr1.reshape((160,140)))
plt.title('mean')
#eigen-face 1-8
for i in range(0,8):
    plt.subplot(3,3,i+1)
    Wtr1=float2uint8(Wtr[:,i])
    plt.imshow(Wtr1.reshape((160,140))) 
    plt.title('Eigenface'+str(i+1))
fig.subplots_adjust(hspace=0.6)
plt.savefig("Eigenfaces.png")
plt.show()
############LDA#######
'''3'''
### LDA data####
LDtr=data_new(90,Wtr,trdata,mtr)
LDte=data_new(90,Wtr,tedata,mtr)
LDAW, Centers, classLabels=myLDA(LDtr,trlabel)
LDAtr=np.dot(LDAW.T,LDtr)
LDAte=np.dot(LDAW.T,LDte)
#the class-mean of LDA is: Centers###
LDAcm=confusionmatrix(Centers,LDAte)
AR_lda=np.trace(LDAcm)/sum(sum(LDAcm))
print('the cofusion matrix of LDA is:\n',LDAcm)
print('the accurate of LDA is:',round(AR_lda,3))
##############LDA plot##
'''4'''
Cp=np.dot(LDAW,Centers)
tiletime=Cp.shape[1]
Cr=np.dot(Wtr[:,:90],Cp)+np.tile(mtr, (tiletime,1)).T
##plot center##
fig = plt.figure()
plt.gray()
for i in range(0,10):
    plt.subplot(2,5,i+1)
    Wtr1=float2uint8(Cr[:,i])
    plt.imshow(Wtr1.reshape((160,140))) 
    plt.title('Center'+str(i+1))
fig.subplots_adjust(wspace=0.5)
plt.savefig("Centerfaces.png")
plt.show()
###################
'''5'''
'''alpha=0.5'''
fuste=fusiondata(te,LDAte,0.5)
# the fusion-class mean of 0.5PCA+0.5LDA#
fuZ=fusiondata(Z,Centers,0.5)
fusioncm=confusionmatrix(fuZ,fuste)
AR_fus=np.trace(fusioncm)/sum(sum(fusioncm))
print('the cofusion matrix of fusion is:\n',fusioncm)
print('the accurate of fusion is:',round(AR_fus,3))
########################
'''6'''
accuracy=np.zeros((9,1))
alpha=np.zeros((9,1))
for i in range(0,9):
    alpha[i]=0.1*(i+1)
    fuste1=fusiondata(te,LDAte,alpha[i])
    fuZ1=fusiondata(Z,Centers,alpha[i])
    fusioncm1=confusionmatrix(fuZ1,fuste1)
    accuracy[i]=round(np.trace(fusioncm1)/sum(sum(fusioncm1)),3)
print(accuracy, alpha)
fig = plt.figure()
plt.scatter(alpha, accuracy, marker = '+', color = 'c') 
plt.title('accuracy versus alpha')
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.savefig('accuracy-alpha.png')
plt.show()




