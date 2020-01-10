import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from scipy import stats

class log_reg():
    def train_test_split(self,dt,sp=0.6,randomize='False'):   # splitting function
        dt=np.array(dt)
        if randomize=='True':
            dt=list(dt)
            random.shuffle(dt)
            dt=np.array(dt)
        n=dt.shape[1]
        m=dt.shape[0]
        train_dt=dt[:int(sp*m),:]
        test_dt=dt[int(sp*m):,:]
        return train_dt,test_dt
    
    def sigmoid(self,tx):                                # sigmoid function
        return (1/(1+np.exp(-tx)))
    
    def hypo(self):                                      # hypothesis function
        self.hx = []
        self.hx = self.x.dot(self.theta[:-1])
        self.hx = self.hx + self.theta[-1]
        self.hx = np.array(self.sigmoid(self.hx))

    def part_j(self):                                    # function to calculate partial derivative terms
        self.diff = self.hx - self.y
        pj = self.x.T.dot(self.diff)
        w = list(pj)
        w.append(np.sum(self.diff))
        pj = np.array(w)
        return pj*(1/self.m)
    
    def cost(self):                                      # cost/loss function
        self.diff = self.hx - self.y
        return (-1/self.m)*np.sum(self.y*np.log(self.hx) + ((1-self.y)*np.log(1-self.hx)))
    
    def score(self,hhx):                                 # function to calculate accuracy
        yy=self.y.copy()                                 # class wise and even overall
        sc=[]
        for i in np.unique(self.y):
          self.y=yy.copy()
          self.y[self.y!=i]=-10
          val=np.count_nonzero((hhx-self.y)==0)
          print(i,100*(val/np.count_nonzero(yy==i)))
          sc.append((val/np.count_nonzero(yy==i))*100)
        return np.array(sc)

    def learn_curve(self,show='False'):                  # learning curve when 'grad des' is used
        if show=='True':                                 # Error VS no.of iteration graph
            leg = []
            for i in np.unique(self.yy):
                plt.plot(self.learn_x,self.learn_j[i])
                leg.append(i)
            plt.xlabel(" no. of iterations")
            plt.ylabel(" error in classes")
            plt.title(" LEARNING CURVES ")
            plt.legend(leg)
            plt.show()
    
    def learn_curve_mb(self,show='Together'):               # learning curve when 'mini batch grad des' is used
        if show=='Together':                                 # Error per Batch graph
            leg = []
            for i in np.unique(self.y_all):
                plt.plot(list(range(self.n_o_b)),np.array(self.j_p_b)[:,i])
                leg.append(i)
            plt.xlabel(" no. of Batches")
            plt.ylabel(" Error per Batch")
            plt.title(" LEARNING CURVES ")
            plt.legend(leg)
            plt.show()
        if show=='Alone':
            for i in np.unique(self.y_all):
                print("Class ",i)
                plt.plot(list(range(self.n_o_b)),np.array(self.j_p_b)[:,i])
                plt.xlabel(" no. of Batches")
                plt.ylabel(" Error per Batch")
                plt.title(" LEARNING CURVE ")
                plt.show()
    
    def test(self,xt,yt):                                # function to test your model
        self.x=xt
        self.y=yt
        self.m=xt.shape[0]
        pred = np.zeros(self.m)
        pred_val=[]
        self.theta_2d=np.array(self.theta_2d)
        for i in range(self.m):
            v = self.theta_2d[:,:-1].dot(self.x[i])
            v = v + self.theta_2d[:,-1]
            v = np.array(self.sigmoid(v))
            pred_val.append(np.max(v))
            pred[i] = np.argmax(v)
        over_all= self.score(pred)
        #print(over_all)
        return pred,np.array(pred_val),(np.sum(over_all)/len(over_all))
    
    def train(self,x,y,f_c='True',itr=50,lr=0.00001):    # function to train your model
        self.x=x
        self.y=y
        self.itr=itr
        self.lr=lr
        self.n=x.shape[1]
        self.m=x.shape[0]
        if f_c=='True':
            self.theta=np.zeros(self.n+1)
            self.theta_2d=[]
            self.j_p_b=[]
            self.n_o_b=0
        self.n_o_b=self.n_o_b + 1
        self.yy = self.y.copy()
        self.j=[]
        self.learn_j=[]
        for index,i in enumerate(np.unique(self.y)):
            k=0
            learn_j_each=[]
            self.learn_x=[]
            self.y=self.yy.copy()
            self.y[self.y!=i]=-1
            self.y[self.y==i]=0
            self.y=self.y+1
            if f_c=='False':
                self.theta=self.theta_2d[index]
            while k < self.itr:
                self.hypo()
                self.theta = self.theta - (self.lr*self.part_j())
                learn_j_each.append(self.cost())
                self.learn_x.append(k)
                k=k+1
            self.learn_j.append(learn_j_each)
            if f_c=='True':
                self.theta_2d.append(self.theta)
            if f_c=='False':
                self.theta_2d=np.array(self.theta_2d)
                self.theta_2d[index]=self.theta
            self.j.append(self.cost())
        self.j_p_b.append(np.array(self.j))
    
    def mini_batch_grad(self,x_all,y_all,batch_size=200,itr_b=43,lr_b=0.000001):
        self.y_all=y_all                                # used when 'mini batch grad des' has to be applied
        n_loop=x_all.shape[0]/batch_size
        if n_loop!=int(n_loop):
            n_loop=int(n_loop)+1
        for i in range(int(n_loop)):
            if i==0:
                self.train(x_all[:batch_size,:],y_all[:batch_size],itr=itr_b,lr=lr_b)
            if i!=0 and i!=n_loop-1:
                self.train(x_all[batch_size*i:batch_size*(i+1),:],y_all[batch_size*i:batch_size*(i+1)],f_c='False',itr=itr_b,lr=lr_b)
            if i==n_loop-1 and i!=0:
                self.train(x_all[batch_size*i:,:],y_all[batch_size*i:],f_c='False',itr=itr_b,lr=lr_b)



class lin_reg():
    
    def train_test_split(self,dt,sp=0.6,randomize='False'):             # splitting function
        dt=np.array(dt)
        if randomize=='True':
            dt=list(dt)
            random.shuffle(dt)
            dt=np.array(dt)
        n=dt.shape[1]
        m=dt.shape[0]
        train_dt=dt[:int(sp*m),:]
        test_dt=dt[int(sp*m):,:]
        return train_dt,test_dt
    
    def hx_curve(self,show='False'):                                   # hypothesis VS Y graph
        if show=='True':
            plt.scatter(self.y,self.hx)
            plt.xlabel("Values of Y values")
            plt.ylabel("Range of Prediction")
            plt.title(" Graph for Hypothesis VS Y values")
            plt.show()
    
    def detr_thres(self,show='False'):                                 # Accuracy VS Threshold
        if show=='True':
            th=list(range(0,101))
            th=np.array(th)
            th=th/10
            accu=[]
            for i in th:
                accu.append(self.score(i))
            plt.plot(th,accu)
            plt.xlabel("Increasing Threshold Value" )
            plt.ylabel("Accuracy of Prediction")
            plt.title("Curve to Determine Threshold")
            plt.show()

    def score(self,t=4):                                              # function to get accuracy
        self.p=self.y-self.hx                                         # for provided threshold 
        self.p[self.p<-t]=-50
        self.p[self.p>t]=-50
        return (100*((len(self.p)-np.count_nonzero(self.p==-50))/len(self.p)))
    
    def hypo(self):                                                   # function to calculate hypothesis
        self.hx = []
        self.hx = np.sum(self.theta[:-1]*self.x,axis=1)
        self.hx = self.hx + self.theta[-1]
    
    def part_j(self):                                                 # function to calculate partial derivative terms 
        self.diff = self.hx - self.y
        pj = self.x.T.dot(self.diff)
        w = list(pj)
        w.append(np.sum(self.diff))
        pj = np.array(w)
        return pj*(1/self.m)
    
    def cost(self):                                                   # function to calculate cost function
        self.diff = self.hx - self.y
        return ((1/(2*self.m))*np.sum(pow(self.diff,2)))
    
    def learn_curve(self,show='False'):                               # Error VS no. of iterations graph
        if show=='True':
            plt.plot(self.learn_x,self.learn_j)
            plt.xlabel(" no. of iterations")
            plt.ylabel(" error ")
            plt.title(" LEARNING CURVE ")
            plt.show() 
    
    def pred_thres(self,ll=0.1,ul=0.9):                               # function to get an array of predicted values
        predict=[]                                                    # for provided lower and upper limit
        self.hypo()                                                   # and accuracy of your prediction as well
        v1=self.hx-np.modf(self.hx)[1]
        for i in range(self.m):
            if 1-v1[i] <=ll:
                predict.append(int(self.hx[i]+1))
            elif v1[i] <ul:
                predict.append(int(self.hx[i]))
            else:
                predict.append(-1)
        predict=np.array(predict)
        p=self.y-predict
        ac=(np.count_nonzero(p==0)/self.m)*100
        return np.array(predict),ac
    
    def learn_curve_mb(self,show='False'):                            # Error per batch graph when 'mini batch grad des' used    
        if show=='True':
            plt.plot(np.array(list(range(self.n_o_b+1)))[1:],np.array(self.j_p_b)[1:])
            plt.xlabel(" Batch numbers")
            plt.ylabel(" error per batch ")
            plt.title(" LEARNING CURVE ")
            plt.show()
    
    def train(self,x,y,theta=[],itr=1000,lr=0.0000001):               # function to train your model
        self.x=x                                                      # if you want 'grad des' call this directly
        self.y=y
        self.itr=itr
        self.lr=lr
        self.n=x.shape[1]
        self.m=x.shape[0]
        self.theta=theta
        if theta==[]:
            self.theta=list(np.zeros(self.n+1))
            self.n_o_b=0
            self.j_p_b=[]
            self.hypo()
            self.j_p_b.append(self.cost())
        self.n_o_b=self.n_o_b + 1
        self.learn_j=[]
        self.learn_x=[]
        i=0
        while i<self.itr:
            self.hypo()
            self.theta = self.theta - (self.lr*self.part_j())
            self.learn_j.append(self.cost())
            self.learn_x.append(i)
            i = i+1
        j = self.cost()
        self.j_p_b.append(j)
    
    def test(self,xt,yt):                                             # function to test your model
        self.x=xt
        self.y=yt
        self.m=self.x.shape[0]
        self.hypo()
        jt=self.cost()
        return self.hx,jt
    
    def mini_batch_grad(self,x_all,y_all,batch_size=250,itr_b=1000,lr_b=0.00000001):       # use this for 'mini batch grad des'
        n_loop=x_all.shape[0]/batch_size
        if n_loop!=int(n_loop):
            n_loop=int(n_loop)+1
        for i in range(int(n_loop)):
            if i==0:
                self.train(x_all[:batch_size,:],y_all[:batch_size],itr=itr_b,lr=lr_b)
            if i!=0 and i!=n_loop-1:
                self.train(x_all[batch_size*i:batch_size*(i+1),:],y_all[batch_size*i:batch_size*(i+1)],self.theta,itr=itr_b,lr=lr_b)
            if i==n_loop-1 and i!=0:
                self.train(x_all[batch_size*i:,:],y_all[batch_size*i:],self.theta,itr=itr_b,lr=lr_b)
                
    
class knn():
  
  def prediction(self,xt_index):
    dist=np.sqrt(np.sum(np.square(self.x-self.xt[xt_index]),axis=1))
    a=np.array([dist,self.y]).T
    sort_a=[]
    for j,i in enumerate(np.argsort(a[:,0])):
        sort_a.append(a[i])
        if j==self.k-1:
            break
    if self.typ=='reg':
        p=(np.mean(np.array(sort_a)[:,1]))
    if self.typ=='cla':
        p=(stats.mode(np.array(sort_a)[:,1])[0][0])
    return p
  
  def train(self,x,y):
    self.x=x
    self.y=y
    self.m=self.x.shape[0]
    self.n=self.x.shape[1]

  def test(self,xt,yt,k,typ):
    self.xt=xt
    self.yt=yt
    self.k=k
    self.typ=typ
    self.mt=self.xt.shape[0]
    self.nt=self.xt.shape[1]
    self.pred=[]
    for i in range(self.mt):
      self.pred.append(self.prediction(i))
    return self.pred
    
  def score(self):
    v=self.yt-self.pred
    accu=np.count_nonzero(v==0)
    print("accuracy is ",(accu/len(v))*100,"%")
    
  def graph(self):
    i=1
    x_axis=[]
    y_axis=[]
    while i<22:
        u=self.test(self.xt[:100,:],self.yt[:100],i,self.typ)
        v=self.yt[:100]-self.pred
        accu=(np.count_nonzero(v==0)/len(v))*100
        x_axis.append(i)
        y_axis.append(accu)
        i=i+2
    plt.plot(x_axis,y_axis)
    plt.xlabel("K values")
    plt.ylabel("Accuracy")
    plt.title("Varying Accuracy wrt K")
    plt.show()


class neural_n_layer():
  def train_test_split(self,dt,sp=0.6,randomize='False'):             # splitting function
        dt=np.array(dt)
        if randomize=='True':
            dt=list(dt)
            random.shuffle(dt)
            dt=np.array(dt)
        n=dt.shape[1]
        m=dt.shape[0]
        train_dt=dt[:int(sp*m),:]
        test_dt=dt[int(sp*m):,:]
        return train_dt,test_dt

  def sigmoid(self,c):
    return 1/(1+np.exp(-c))
  
  def create_a(self,ar,i):
    z=self.theta[i].dot(ar)
    z=np.array(self.sigmoid(z))
    if i!=self.hl:
      z=np.append(z,1)
    return z
  
  def for_prop(self):
    self.a_2d=[]
    a_0=list(self.x)
    a_0.append(1)
    self.a_2d.append(np.array(a_0))
    for i in range(self.hl+1):
      self.a_2d.append(self.create_a(np.array(self.a_2d)[i],i))

  def initialize(self):
    self.hl=int(input("no. of hidden layers --"))
    self.n_hl=np.array([])
    self.theta=[]
    for i in range(self.hl):
      print("for hidden layer",i+1)
      self.n_hl=np.append(self.n_hl,int(input(" tell no. of nodes --")))
      if i==0:
        self.theta.append(np.zeros((int(self.n_hl[i]),int(self.n+1))))
      elif i!=0:
        self.theta.append(np.zeros((int(self.n_hl[i]),int(self.n_hl[i-1]+1))))
    self.theta.append(np.zeros((len(np.unique(self.y_all)),int(self.n_hl[-1]+1))))
    self.theta=np.array(self.theta)
    for i in range(len(self.theta)):
      for j in range(len(self.theta[i])):
        for k in range(len(self.theta[i][j])):
            self.theta[i][j][k]=random.random()

  def backprop(self):
    sigma=[]
    sigma.append(self.a_2d[-1]-self.y)
    #sigma.append((self.y*np.log(self.a_2d[-1]) + ((1-self.y)*np.log(1-self.a_2d[-1])))/len(self.a_2d[-1]))
    for i in range(self.hl):
      if i==0:
        sigma.append(self.theta[-i-1].T.dot(np.array(sigma)[i])*self.a_2d[-i-2]*(1-self.a_2d[-i-2]))
      elif i!=0:
        sigma.append(self.theta[-i-1].T.dot(np.array(sigma)[i][:-1])*self.a_2d[-i-2]*(1-self.a_2d[-i-2]))
    sigma=np.flipud(np.array(sigma))
    return sigma
  
  def y_det(self,yy):
    for index,i in enumerate(np.unique(self.y_all)):
      if yy==i:
        yy=np.zeros(len(np.unique(self.y_all)))
        yy[index]=1
        return yy
  
  def accuracy(self):
    accu=self.yt-self.pred
    print("accuracy is --",(np.count_nonzero(accu==0)/len(self.yt))*100)

  def train(self,x,y,itr=50,lr=4):
    self.x_all=x
    self.y_all=y
    self.n=self.x_all.shape[1]
    self.m=self.x_all.shape[0]
    self.itr=itr
    self.lr=lr
    self.initialize()
    k=0
    self.cj=[]
    self.learn_x=[]
    while k<self.itr:  
      self.delta=self.theta-self.theta
      j=[]
      for eg in range(self.m):
        self.x=self.x_all[eg]
        self.x=(self.x-np.mean(self.x))/(np.max(self.x)-np.min(self.x))
        self.y=self.y_det(self.y_all[eg])
        self.for_prop()
        j.append(np.sum((self.y*np.log(self.a_2d[-1])) + ((1-self.y)*np.log(1-self.a_2d[-1]))))
        s=self.backprop()
        #print(eg,"  ",self.a_2d[1],self.a_2d[2])
        #print(s)
        d=[]
        for l in range(self.hl+1):
          if l!=self.hl:
            a_s=np.array([self.a_2d[l]])*np.array([s[l][:-1]]).T
          elif l==self.hl:
            a_s=np.array([self.a_2d[l]])*np.array([s[l]]).T
          d.append(a_s)
        d=np.array(d)
        self.delta=self.delta + d
      cj=-(np.sum(np.array(j)))/self.m
      self.cj.append(cj)
      self.learn_x.append(k+1)
      #print(k,"   ",cj)
      self.delta=self.delta/self.m
      '''blank=[]
      for h in range(len(self.theta)):
          e=self.theta[h].copy()
          e[:,-1]=0
          blank.append(e)
      blank=np.array(blank)
      self.delta += 10*blank'''
      #print(self.delta)
      self.theta=self.theta-(self.lr*self.delta)
      #plt.scatter(k+1,cj,color='red')
      k=k+1
    print("loss value is --",cj)
    #plt.show()
    
  def learn_curve(self,show='False'):
    if show=='True':                                
            plt.plot(self.learn_x,self.cj)
            plt.xlabel(" no. of iterations")
            plt.ylabel(" error ")
            plt.title(" LEARNING CURVE ")
            plt.show()
  
  def test(self,xt,yt):
    self.yt=yt
    self.pred=np.array([])
    #print(self.theta)
    for i in range(xt.shape[0]):
      self.x=xt[i]
      self.x=(self.x-np.mean(self.x))/(np.max(self.x)-np.min(self.x))
      self.for_prop()
      self.pred=np.append(self.pred,np.unique(self.y_all)[np.argmax(self.a_2d[-1])])
      #print(self.a_2d[-1])
    return self.pred

 
def convert_data(data,concerned_class,threshold=5):
  data[:,1:][data[:,1:]>=threshold]=1
  data[:,0][data[:,0]!=concerned_class]=-1
  data[:,0][data[:,0]==concerned_class]=0
  data[:,0]+=1
  return data

class decision_tree():
  def __init__(self,data):
    self.comp_data=data
    self.i=1

  def calc_gini(self,data):
    if data.size==0:
      return 0
    d=data[:,0]
    p = np.count_nonzero(d==1)/len(d)
    return (p*p)+((1-p)*(1-p))

  def gini_split_val(self,data,col,split='False'):
    d=data[:,col]
    d1=[]
    d2=[]
    for i in range(data.shape[0]):
      if d[i]==0:
        d1.append(data[i,:])
      elif d[i]==1:
        d2.append(data[i,:])
    d1=np.array(d1)
    d2=np.array(d2)
    gini1=self.calc_gini(d1)
    gini2=self.calc_gini(d2)
    if split=='True':
      if d1.size==0:
        return d1,np.delete(d2,[col],1)
      elif d2.size==0:
        return np.delete(d1,[col],1),d2
      else:
        return np.delete(d1,[col],1),np.delete(d2,[col],1)
    if gini1==0:
      return gini2
    elif gini2==0:
      return gini1
    #check if both gini are not 0
    else :
      return ((d1.shape[0]/len(d))*gini1) + ((d2.shape[0]/len(d))*gini2)
    
  def choosing_col(self,data):
    gini_vals=[]
    for i in range(data.shape[1]-1):
      gini_vals.append(self.gini_split_val(data,i+1))
    return np.argmax(np.array(gini_vals))+1
  
  def build_tree(self,dataset,parent_class='None'):
    #print(self.i)
    self.i+=1
    #print(dataset)
    if dataset.size==0:
      return np.unique(self.comp_data[:,0])[np.argmax(np.unique(self.comp_data[:,0],return_counts=True)[1])]
    elif len(np.unique(dataset[:,0]))<=1:
      return dataset[0][0]
    elif dataset.shape==(dataset.shape[0],1):
      return parent_class
    else:
      parent_class= np.unique(dataset[:,0])[np.argmax(np.unique(dataset[:,0],return_counts=True)[1])]
      attr_no = self.choosing_col(dataset)
      tree = {attr_no:{}}
      #for value in np.unique(dataset[attr_no]):
      for value in range(2):
        t = self.gini_split_val(dataset,attr_no,split='True')
        subtree= self.build_tree(t[value],parent_class)
        tree[attr_no][value]=subtree
      return tree
  
  def prediction(self,query,tree,default_val='None'):
    if default_val=='None':
      default_val=np.unique(self.comp_data[:,0])[np.argmax(np.unique(self.comp_data[:,0],return_counts=True)[1])]
    for i in list(query.keys()):
      if i in list(tree.keys()):
        try:
          out= tree[i][query[i]]
        except:
          return default_val
        out = tree[i][query[i]]
        if isinstance(out,dict):
          return self.prediction(query,out)
        else:
          return out
  
  def test(self,data,tree):
    test_data=data[:,1:]
    y=data[:,0]
    count=0
    for i in range(data.shape[0]):
      test_query=dict(enumerate(test_data[i,:]))
      pred=self.prediction(test_query,tree)
      if pred==y[i]:
        count+=1
    print((count/len(y))*100)

  
