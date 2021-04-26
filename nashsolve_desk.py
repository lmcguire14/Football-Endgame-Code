# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:16:33 2021

@author: m214398
"""

import numpy as np
import scipy

def nash_equil(payoff_matrix):
    try:
        sample = payoff_matrix
        A_ub = np.ones((np.shape(sample)[0],np.shape(sample)[1]+1))
        A_ub[:,:-1] = -sample[:,:] 
        c = np.zeros(np.shape(A_ub)[1])
        c[-1] = -1
        row_num = np.shape(A_ub)[0]
        b_ub = np.zeros((row_num,1))
        Aeq = np.transpose(1 - (np.zeros((np.shape(sample)[1]+1,1))))
        Aeq[0][-1] = 0
        beq = np.ones((1,1)) 
        solution = (((scipy.optimize.linprog(c, A_ub, b_ub, Aeq, beq, [0,1]))))
        success_prob = solution['fun']
        strategy_off =  solution['x']
        if not isinstance(strategy_off, np.ndarray):
            strategy_def = (1/((np.shape(sample)[1]-3)))* (np.ones((np.shape(sample)[1],1)))
            strategy_off[-3:-1] = 0
            success_prob = -np.average(payoff_matrix[:(np.shape(sample)[1]) - 1,:])
    #        print("ALARM OFFENSE")
    #        print(-success_prob)
        A_ub = -1 * np.ones((np.shape(sample)[0]+1,np.shape(sample)[1]))
        A_ub[:-1,:] = sample[:,:] 
        A_ub = np.transpose(A_ub)
        c = np.zeros(np.shape(sample)[0] + 1)
        c[-1] = 1 
        row_num = np.shape(A_ub)[0]
        b_ub = np.zeros((row_num,1))
        Aeq = np.transpose(1 - (np.zeros((np.shape(sample)[0] + 1,1))))
        Aeq[0][-1] = 0
        beq = np.ones((1,1))
        # print(A_ub)
        solution = ((scipy.optimize.linprog(c, A_ub, b_ub, Aeq, beq, [0,1])))
        strategy_def =  solution['x']
        if not isinstance(strategy_def, np.ndarray):
            strategy_def = (1/((np.shape(sample)[0])))* (np.ones((np.shape(sample)[0],1)))
            success_prob = np.average(payoff_matrix[:(np.shape(sample)[1]) - 1,:])
            success_prob = -(success_prob)
    #        print("ALARM DEFENSE")
            success_prob = round(success_prob, 5)
    #        print(-success_prob)
            return (-success_prob), strategy_off[0:-1], strategy_def[0:]
        else:
            return (-success_prob), strategy_off[0:-1], strategy_def[0:-1] 
    except ValueError:
        strategy_def = (1/((np.shape(sample)[0])))* (np.ones((np.shape(sample)[0],1)))
        success_prob = np.average(payoff_matrix[:(np.shape(sample)[1]) - 1,:])
        success_prob = -(success_prob)
    #   print("ALARM DEFENSE")
        success_prob = round(success_prob, 5)
    #   print(-success_prob)
        return (-success_prob), strategy_off[0:-1], strategy_def[0:]