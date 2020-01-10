import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

class k_means():
  def initialize(self):
    self.cent_index=np.zeros(self.k)
    self.cent_x=[]
    self.cent_y=[]
    for i in range(self.k):
      self.cent_index[i]=random.randint(0,self.m-1)
      self.cent_x.append(self.x[int(self.cent_index[i])])
      self.cent_y.append(self.y[int(self.cent_index[i])])
    self.cent_x=np.array(self.cent_x)
    self.cent_y=np.array(self.cent_y)
  
  def distance(self,xc):
    dist=np.sqrt(np.sum(pow(self.cent_x-xc,2),axis=1))
    cent_no=np.argmin(dist)
    label=self.cent_y[cent_no]
    return label,cent_no

  def train(self,x,y,itr,k):
    self.x=x
    self.y=y
    self.itr=itr
    self.k=k
    self.m=self.x.shape[0]
    self.n=self.x.shape[1]
    t=0
    self.initialize()
    while t <self.itr:
      self.label_array=[]
      centroid_list=[]
      for j in range(self.k):
        centroid_list.append(np.array([]))
      centroid_list=np.array(centroid_list)
      for i in range(self.m):
        l,c=self.distance(self.x[i])
        centroid_list[c]=np.append(centroid_list[c],i)
        self.label_array.append(l)
      for i in range(self.k):
        d=[]
        for j in centroid_list[i]:
          d.append(self.x[j])
        d=np.array(d)
        #print(d)
        self.cent_x[i]=np.mean(d,axis=0)
      t=t+1
    print((np.count_nonzero((np.array(self.label_array)-self.y)==0)/len(self.y))*100)
  
  def accuracy(self):
    print((np.count_nonzero((self.yt-self.pred)==0)/len(self.yt))*100)

  def test(self,xt,yt):
    self.yt=yt
    self.x_test = xt
    self.pred=[]
    for i in range(xt.shape[0]):
       l,c=self.distance(xt[i])
       self.pred.append(l)
    return np.array(self.pred)

  def plot(self,show='Test'):
    if show=='Test':
      plt.scatter(self.x_test[:,0],self.x_test[:,1],color='orange')
      plt.scatter(self.cent_x[:,0],self.cent_x[:,1],color='red')
      plt.show()
    if show=='Train':
      plt.scatter(self.x[:,0],self.x[:,1],color='orange')
      plt.scatter(self.cent_x[:,0],self.cent_x[:,1],color='red')
      plt.show()

