# particle swarm optimization article - restarts PSO 
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import functions
from pandas import DataFrame
from restarts_PSO import ParticleSwarmOptimizer, Particle


# gbest_value = np.zeros((30 , 29))
for run in range(0 ,30):
    if not os.path.exists("output_"+ str(run) +"/gbest"):
        os.makedirs("output_"+ str(run) +"/gbest")    
    for func_num in range(0 , 1):
        restarts_pso = ParticleSwarmOptimizer(func_num , run)
        gbest= restarts_pso.optimize()

        # gbest_value[run][func_num] = gbest[-1] 

        print('Function', func_num, 'is processed:')

        if not os.path.exists("output_"+ str(run) +"/gbset_val"):
            os.makedirs("output_"+ str(run) +"/gbset_val")
        with open("output_"+ str(run) +"/gbset_val/gbest_" + str(func_num) + ".csv", 'a+') as f:
            f.write(','.join([str(item) for item in gbest]))

        plt.plot(gbest)
        plt.xlabel('iteration')
        plt.ylabel('gbest')
        plt.title('new Method')
        plt.savefig("output_"+ str(run) +"/gbest/func_" + str(func_num) + ".jpg")
        plt.clf()

        with open("final_result.csv", 'a+') as f:
            if func_num > 0:
                f.write(','+ str(gbest[-1]))
            else:
                f.write(str(gbest[-1]))

    with open("final_result.csv", 'a+') as f:
        f.write('\n')
