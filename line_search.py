import random
import functions
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas import DataFrame
import statistics
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
#Hyper parameters 
DIMENSION = 30 # Size of the problem
PASS_DEPTH = 7

class LandscapeAnalysisTools:
    def __init__(self, func_num ,run):
        self.run = run
        self.f = func_num
        self.cec_functions = functions.CEC_functions(30)                
        return

    def cic13fun(self, x):
        arr_x = np.array(x).astype(float)
        if (self.f == 0):
            sq = arr_x ** 2
            return 10*len(arr_x) + np.sum(sq - 10*np.cos((2*np.pi*arr_x).astype(float)) )
        else:
            return self.cec_functions.Y(arr_x, self.f)

    def linesearch(self, m, start = -100, stop = 100): # Ls stands for linesearch
        
        a = np.array([None for i in range(30)])
        for j in range(30):
            if j != m:
                a[j] = random.uniform(start,stop)
        a[m] = start
        b = np.array([None for i in range(30)])
        for j in range(30):
            if j!= m:
                b[j] = random.uniform(start,stop)      
        b[m] = stop

        dist = np.linalg.norm(a-b)   # Euclidean distance between two vectors of a and b
        a = a.astype(float)
        b = b.astype(float)
        alpha = np.array([x/999 for x in range(1000)])
        X = [a + al * (b - a) for al in alpha]
        X = np.array(X)
        X = X.astype(float)
        cost = np.array([self.cic13fun((x)) for x in X])
        min_cost = np.min(cost)
        index = np.argmin(min_cost)
        min_pos = X[index]
        return  alpha, cost, min_cost , min_pos

    def Minimoptima(self, Y_,X_):
        w=[]
        x=[]
        d=len(X_)
        for i in range(1, d-1):
            # if i == 0:
            #     if Y_[0] < Y_[1] :
            #         w.append(Y_[i])
            #         x.append(X_[i])
            # elif i == d - 1: 
            #     if Y_[d-1] < Y_[d-2]:
            #         w.append(Y_[i]) 
            #         x.append(X_[i])
            if (Y_[i] <Y_[i-1] and Y_[i+1] > Y_[i]):
                    w.append(Y_[i])
                    x.append(X_[i])

        return x , np.array(w)


    def plot_min(self, min_Xi, min_Yi, p, dim, alpha_, Y1):
        plt.xlim(0, 1)
        # plt.ylim(0,750)
        plt.plot(alpha_ , Y1)
        plt.scatter(min_Xi , min_Yi , marker='o' , c='red')
        if not os.path.exists("output_"+ str(self.run) +"/line_search/func_" + str(self.f) + "/dim_" + str(dim) ):
            os.makedirs("output_"+ str(self.run) +"/line_search/func_" + str(self.f) + "/dim_" + str(dim)) 
        plt.savefig("output_"+ str(self.run) +"/line_search/func_" + str(self.f) + "/dim_" + str(dim) +"/graph_" + str(p) +".jpg")
        plt.clf()
    
    def depth_report(self):

        min_pos_list = []
        min_value_list =[]

        if not os.path.exists("output_"+ str(self.run) +"/min-dis"):
            os.makedirs("output_"+ str(self.run) +"/min-dis") 
        if not os.path.exists("output_"+ str(self.run) +"/depth"):
            os.makedirs("output_"+ str(self.run) +"/depth") 

        Depth = [0] * DIMENSION
        min_dif_avg = [0] * DIMENSION   # will be used in perturbation
        # Dimension loop
        for dim in range(DIMENSION):
            min_dif = []
            if (self.f == 0):
                alpha_, Yi, min_value , min_pos = self.linesearch(dim, -5.12, 5.12)
            else:
                alpha_, Yi, min_value , min_pos = self.linesearch(dim)
            min_pos_list.append(min_pos)
            min_value_list.append(min_value)
            print("Dimension ", dim, " is processed")

            #Initialization
            Y1, min_Yi, min_Xi  = Yi , Yi, alpha_

            #Main loop
            for p in range(PASS_DEPTH):
                min_Xi , min_Yi = self.Minimoptima(min_Yi, min_Xi)
                d = len(min_Xi)

                if d <= 1:
                    # counting path where d==1
                    Depth[dim] = p
                    if p == 0:
                        min_dif.append(1)
                        min_dif_avg[dim] = 1

                else:
                    # counting oath where d==2
                    if p == 0:
                        for i in range(d - 1):
                            min_dif.append(min_Xi[i+1] - min_Xi[i])

                        min_dif_avg[dim] = np.min(min_dif)
                        
                self.plot_min(min_Xi, min_Yi, p, dim, alpha_ , Y1)

                if len(min_Xi) == 1:    
                    break

            with open("output_"+ str(self.run) +"/min-dis/dis_" + str(self.f) + ".csv", 'a+') as f:
                f.write('DIM ' + str(dim) + ',' + ','.join([str(item) for item in min_dif]) + '\n')

        # we said, if more than half been recognized as unimodal, consider it as unimodal and gave the same basin size
        # if (min_dif_avg.count(1) > DIMENSION / 2):
        #     min_dif_avg = [1] * DIMENSION

        Distance = {'Dimension': range(DIMENSION), 'Distances Average': min_dif_avg , 'Depth': Depth, 'Minimum Position': min_pos_list, 'Minimum Value':min_value_list}
        result = DataFrame(Distance)
        result[['Distances Average', 'Depth', 'Minimum Value']].to_csv('output_'+ str(self.run) +'/depth/func_' + str(self.f) + '.csv')
        return min_dif_avg, min_pos_list, min_value_list

    def Maxioptima(self, Y_,X_):
        w=[]
        x=[]
        d=len(X_)
        for i in range(d):
            if i == 0:
                if Y_[0] > Y_[1] :
                    w.append(Y_[i])
                    x.append(X_[i])
            elif i == d - 1: 
                if Y_[d-1] > Y_[d-2]:
                    w.append(Y_[i]) 
                    x.append(X_[i])
            elif (Y_[i] > Y_[i-1] and Y_[i+1] < Y_[i]):
                    w.append(Y_[i])
                    x.append(X_[i])

        return x , np.array(w)
