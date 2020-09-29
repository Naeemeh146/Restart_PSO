# particle swarm optimization article - restarts PSO 
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from line_search import LandscapeAnalysisTools
import statistics
import functions
import os
#Hyper parameters 
w = 0.729844 # Inertia weight to prevent velocities becoming too large
c1 = 2.05 * w # Scaling co-efficient on the Social component
c2 = 2.05 * w # Scaling co-efficient on the Cognitive component
dimension = 30 # Size of the problem
evalspPerDim = 10000
swarmsize = 50    #population size
maxevals = evalspPerDim * dimension
iteration = math.floor(maxevals/swarmsize)


# This class contains the code of the Particles in the swarm
class Particle:

    def __init__(self, fun_num, run, RangeF = 100):
        # PSO parameters -- D. Bratton and J. Kennedy, "Defining a standard for particle swarm optimization," IEEE SIS, 2007, pp. 120â€“127.
        self.pos  = np.random.uniform(low= -1 * RangeF, high=RangeF, size=dimension)
        self.pbest = self.pos
        self.velocity = np.array([0] * dimension) # zero for initial velocity -- A. Engelbrecht, "Particle Swarm Optimization: Velocity Initialization," IEEE CEC, 2012, pp. 70-77.
        self.Vmin, self.Vmax = -1 * RangeF, RangeF
        self.Xmin, self.Xmax = -1 * RangeF, RangeF
        self.run =run
        self.f = fun_num
        self.cec_functions = functions.CEC_functions(30) 
        self.val = self.cic13fun(self.pos)
        self.pbest_val = self.val
        return

    def cic13fun(self, x):
        arr_x = np.array(x).astype(float)
        if (self.f == 0):
            sq = arr_x ** 2
            return 10*len(arr_x) + np.sum(sq - 10*np.cos((2*np.pi*arr_x).astype(float)) )
        else:
            return self.cec_functions.Y(arr_x, self.f)

    def update_velocities(self, Xlbest):
        r1 = np.random.uniform(size= dimension)
        r2 = np.random.uniform(size= dimension)
        social = c1 * r1 * (Xlbest - self.pos)
        cognitive = c2 * r2 * (self.pbest - self.pos)
        self.velocity = (w * self.velocity) + social + cognitive
        self.velocity = np.clip(self.velocity, self.Vmin, self.Vmax)
        return

    def update_position(self):
        self.pos = self.pos + self.velocity
        for i in range(dimension):
            # clamp position: all positions need to be inside of the boundary, reflect them in cased they are out of the boundary
            # based on: S. Helwig, J. Branke, and S. Mostaghim, "Experimental Analysis of Bound Handling Techniques in Particle Swarm Optimization," IEEE TEC: 17(2), 2013, pp. 259-271
            if self.pos[i] < self.Xmin:
                Lrelectionamount = self.Xmin - self.pos[i] # Less than lower bounds
                self.pos[i] = self.Xmin + Lrelectionamount  # reflect
                self.velocity[i] = 0    # set velocity for reflected particles to zero
            if self.pos[i] > self.Xmax:
                Urelectionamount =  self.Xmax - self.pos[i] # Higher than upper bounds
                self.pos[i] = self.Xmax + Urelectionamount  # reflect
                self.velocity[i] = 0   # set velocity for reflected particles to zero
        self.val = self.cic13fun(self.pos)
        #Update pbest
        if self.cic13fun(self.pos) < self.pbest_val: 
            self.pbest, self.pbest_val = self.pos, self.cic13fun(self.pos)
        return

# This class contains the particle swarm optimization algorithm   
class ParticleSwarmOptimizer:
    
    def __init__(self,fun_num , run):
        self.swarm = []
        self.fun_num = fun_num
        self.run = run
        #Initial particles (position and velocity)
        for _ in range(swarmsize):
            if (fun_num == 0):
                p = Particle(fun_num, run ,5.12)
                self.rangeX = 10.24  
            else:
                p = Particle(fun_num , run)   
                self.rangeX = 200   
            self.swarm.append(p)
        return

    def Lbest(self, pindex):
        adj_p = [self.swarm[(pindex-1 + swarmsize) % swarmsize], self.swarm[pindex], self.swarm[(pindex+1) % swarmsize]] #index wise adjacent particles
        adj_c = [adj_p[0].pbest_val, adj_p[1].pbest_val, adj_p[2].pbest_val]
        lbest = np.argmin(adj_c)
        return adj_p[lbest]

    def Gbest(self):
        pbest_list = [p.pbest_val for p in self.swarm]
        gbest = np.min(pbest_list)
        gbest_pos = self.swarm[np.argmin(pbest_list)].pbest
        return gbest , gbest_pos

    def optimize(self): 
        resetThreshold = int(iteration / 10)
        gbest = []
        lat = LandscapeAnalysisTools(self.fun_num , self.run)
        min_dif_avg, min_pos, min_value = lat.depth_report()
        min_dif_avg = np.array(min_dif_avg)
        p_range = min_dif_avg
        samples = random.sample(range(swarmsize), k=dimension)  # select a list of size 30 randomly from self.swarm without replacement

        if not os.path.exists("output_"+ str(self.run) +"/Initial_positions"):
            os.makedirs("output_"+ str(self.run) +"/Initial_positions")
        #Assigning pos, pbest to 30 particles selected from swarm
        # lbest = []
        j = 0
        initial_pos =[]
        for i in range(swarmsize):
            if (i in samples):  
                self.swarm[i].pos = min_pos[j]
                initial_pos.append(min_pos[j])
                self.swarm[i].pbest = min_pos[j]
                self.swarm[i].pbest_val = min_value[j]
                j = j + 1

        for p in self.swarm:
            with open("output_"+ str(self.run) +"/Initial_positions/" + str(self.fun_num) + ".csv", 'a+') as f:
                f.write(','.join([str(item) for item in p.pos]) + '\n')
        
        #Initialization of lbest for all particles
        lbest = []
        for j in range(swarmsize):
            lbest.append(self.Lbest(j).pbest)  # I am not sure about this one?!!

        # keeping initializing points as part of wanted information by CEC


        for i in range(601, iteration):
            # print('iteration', i, 'is done!')
            #Check for iteration threshold
            if ((i % resetThreshold) == 0 and i < (iteration * 0.95)):
                print('-----------reset threshold-----------  ', i)
                pval = []
                for p in self.swarm:
                    if (i < (iteration * 0.85)):
                        p.pos = p.pbest + np.array([np.random.uniform(low= (-0.75)* p_range[i] * self.rangeX, high= (+0.75)* p_range[i] * self.rangeX) \
                        for i in range(dimension)]) 
    
                    p.pbest = p.pos
                    p.pbest_val = p.cic13fun(p.pbest)
                    pval.append(p.pbest_val)

               
                if ((i > 2*resetThreshold)):
                    j = np.argmax(pval)
                    # np.random.choice(self.swarm).pbest = gbest[-1]
                    self.swarm[j].pbest = gbest[-resetThreshold-1][1]
                    self.swarm[j].pbest_val = gbest[-resetThreshold-1][0]

                if ((i == resetThreshold)):
                    j = np.argmax(pval)
                    # np.random.choice(self.swarm).pbest = gbest[-1]
                    self.swarm[j].pbest = gbest[-resetThreshold][1]
                    self.swarm[j].pbest_val = gbest[-resetThreshold][0]

                if int(i / resetThreshold) % 2 == 0:
                    p_range = p_range * 0.95
                    # random.shuffle(self.swarm)
                    

            
            #Update of velocity and position
            for j, p in enumerate(self.swarm):
                p.update_velocities(lbest[j])
                p.update_position()


            #Update of lbest
            lbest = []
            for j in range(swarmsize):
                p = self.Lbest(j)
                lbest.append(p.pbest)

            #Update gbest
            gbest.append(self.Gbest())

        print(self.Gbest())

        return [x[0] for x in gbest]



            
    


           



    

        