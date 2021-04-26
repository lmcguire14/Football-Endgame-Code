# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:24:53 2021

@author: arris
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:59:13 2020

@author: m214398
"""

import numpy as np 
from scipy.stats import logistic
import time as tme
from nashsolve import nash_equil


#Deterministic array we started this entire process with
#kept for ease of seeing the breakdown in row/column decisions
S_with_names = (np.array([[' ',   'RunInside','RunOutside', 'PassShort','PassDeep'],
                          ['Blitz',1         ,2           ,5           ,6],
                          ['Man'  ,4         ,4           ,4           ,3],
                          ['Zone' ,4         ,4           ,2           ,4],
                          ['DZone',5         ,4           ,3           ,1]]))


################### ARRAYS ###################################################
#All of the below data (unless stated otherwise) was derived by scraping the
#NFLFastR data and fitting distributions or by calculating parameters

#mu is the location parameter for the logistic fit of each play type interaction
global mu_array_n
mu_array_n = (np.array([[3.389129*(1-.20), 4.327093*(1-.20), 8.453942*(1+.10), 26.50913*(1-.10)],
                    [3.389129*(1+.10), 4.327093*(1+.10), 8.453942*(1-.05), 26.50913*(1+.05)],
                    [3.389129*(1-.10), 4.327093*(1-.05), 8.453942*(1+.00), 26.50913*(1+.00)],
                    [3.389129*(1+.15), 4.327093*(1+.10), 8.453942*(1+.10), 26.50913*(1-.15)]]))

#s is the scale parameter for the logistic fit of each play type interaction
global s_array_n
s_array_n = (np.array([[2.512748*(1-.10), 3.438559*(1+.10), 3.508151*(1+.05), 6.39072*(1+.20)],
                       [2.512748*(1+.00), 3.438559*(1+.00), 3.508151*(1+.05), 6.39072*(1+.05)],
                       [2.512748*(1-.05), 3.438559*(1+.00), 3.508151*(1+.00), 6.39072*(1-.075)],
                       [2.512748*(1+.05), 3.438559*(1-.05), 3.508151*(1-.05), 6.39072*(1-.10)]]))

#incompletion probability for each play type
global incomplete_array_n
incomplete_array_n = (np.array([[0, 0 ,(1-0.6785083)*(1+ 0.15), (1-0.4053064)*(1+.20)],
                                 [0, 0 ,(1-0.6785083)*(1+.075), (1-0.4053064)*(1-.05)],
                                 [0, 0 ,(1-0.6785083)*(1+.125), (1-0.4053064)*(1+.075)],
                                 [0, 0 ,(1-0.6785083)*(1-.150), (1-0.4053064)*(1+.20)]]))  

#turnover probabilitiy for each playtype
global turnover_array_n
turnover_array_n = (np.array([[0.005679929*(1+.10), 0.005151475*(1+.10), 0.02163225*(1+.075), 0.06206500*(1+.125)],
                              [0.005679929*(1+.00), 0.005151475*(1+.00), 0.02163225*(1+.000), 0.06206500*(1+.00)],
                              [0.005679929*(1+.075), 0.005151475*(1+.05), 0.02163225*(1+.10), 0.06206500*(1+.05)],
                              [0.005679929*(1+.02), 0.005151475*(1+.05), 0.02163225*(1-.10), 0.06206500*(1+.10)]]))

#probability each offensive play type takes a certain amount of time
#Note: for pass plays, this is the time for completions
#1st row is RI, 2nd row RO, 3rd is PS, 4th is PD
#The first item in the list represents the probability the play takes 7 sec
#subsequent entries represent the probability it takes 14,21,28,etc. sec
global time_array_n
time_array_n = (np.array([[0.060598138, 0.033407411, 0.024343835, 0.063532180, 0.196392000, 0.436097433, 0.179993319, 0.005635685],
                       [0.061562945, 0.049529553, 0.033587403, 0.094089399, 0.194712008, 0.362369824, 0.195270402, 0.008878466],
                       [0.090924360, 0.067232145, 0.064638038, 0.135565385, 0.200343392, 0.291062464, 0.142358584, 0.007875632],
                       [0.10941424, 0.15122514, 0.08830517, 0.12753682, 0.16242449, 0.19547954, 0.14817077, 0.01744383]]))

#same structure as the above array, just examines the time breakdown for incomplete passes
global inc_time_array_n
inc_time_array_n = (np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0],
                       [8.939146e-01, 9.608753e-02, 7.243166e-03, 1.386989e-03, 4.430660e-04, 5.393847e-04, 3.274836e-04, 5.779122e-05],
                       [0.10941424, 0.15122514, 0.08830517, 0.12753682, 0.16242449, 0.19547954, 0.14817077, 0.01744383]]))

###############################################################################

################### 2 MINUTE SPECIFIC ARRAYS ##################################
#mu is the location parameter for the logistic fit of each play type interaction
global mu_array_hu
mu_array_hu = (np.array([[2.718662*(1-.20), 5.538990*(1-.20), 8.802674*(1+.10), 24.724537*(1-.10)],
                         [2.718662*(1+.10), 5.538990*(1+.10), 8.802674*(1-.05), 24.724537*(1+.05)],
                         [2.718662*(1-.10), 5.538990*(1-.05), 8.802674*(1+.00), 24.724537*(1+.00)],
                         [2.718662*(1+.15), 5.538990*(1+.10), 8.802674*(1+.10), 24.724537*(1-.15)]]))

#s is the scale parameter for the logistic fit of each play type interaction
global s_array_hu
s_array_hu = (np.array([[2.174490*(1-.10), 3.394831*(1+.10), 3.209495*(1+.05), 4.903618*(1+.20)],
                        [2.174490*(1+.00), 3.394831*(1+.00), 3.209495*(1+.05), 4.903618*(1+.05)],
                        [2.174490*(1-.05), 3.394831*(1+.00), 3.209495*(1+.00), 4.903618*(1-.075)],
                        [2.174490*(1+.05), 3.394831*(1-.05), 3.209495*(1-.05), 4.903618*(1-.10)]]))

#incompletion probability for each play type
global incomplete_array_hu
incomplete_array_hu = (np.array([[0, 0 ,(1-0.6114234)*(1+ .150), (1-0.3493899)*(1+.20)],
                                  [0, 0 ,(1-0.6114234)*(1+.075), (1-0.3493899)*(1-.05)],
                                  [0, 0 ,(1-0.6114234)*(1+.125), (1-0.3493899)*(1+.075)],
                                  [0, 0 ,(1-0.6114234)*(1-.150), (1-0.3493899)*(1+.20)]]))  

#turnover probabilitiy for each playtype
global turnover_array_hu    
turnover_array_hu = (np.array([[0.006479482*(1+.10), 0.004319654*(1+.10) , 0.02855858*(1+.075), 0.1053308*(1+.125)],
                               [0.006479482*(1+.00), 0.004319654*(1+.00) , 0.02855858*(1+.00), 0.1053308*(1+.00)],
                               [0.006479482*(1+.075), 0.004319654*(1+.05) , 0.02855858*(1+.10), 0.1053308*(1+.05)],
                               [0.006479482*(1+.02), 0.004319654*(1+.05) , 0.02855858*(1-.10), 0.1053308*(1+.10)]]))

#probability each offensive play type takes a certain amount of time
#Note: for pass plays, this is the time for completions
#1st row is RI, 2nd row RO, 3rd is PS, 4th is PD
#The first item in the list represents the probability the play takes 7 sec
#subsequent entries represent the probability it takes 14,21,28,etc. sec
global time_array_hu
time_array_hu = (np.array([[0.42352941, 0.16705882, 0.07529412, 0.11764706, 0.04470588, 0.09882353, 0.07294118, 0],
                       [0.399209486, 0.312252964, 0.063241107, 0.102766798, 0.027667984, 0.031620553, 0.055335968, 0.007905138],
                       [0.357506860, 0.172873383, 0.199921599, 0.178753430, 0.051352411, 0.024696198, 0.013328107, 0.001568013],
                       [0.348249027, 0.210116732, 0.206225681, 0.143968872, 0.058365759, 0.011673152, 0.019455253, 0.001945525]]))

#same structure as the above array, just examines the time breakdown for incomplete passes
global inc_time_array_hu
inc_time_array_hu = (np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0],
                       [0.9336283186, 0.0556257901, 0.0069532238, 0.0031605563, 0.0006321113, 0, 0, 0],
                       [0.832547170, 0.158018868, 0.008254717, 0.001179245, 0, 0, 0, 0]]))


global play_count
play_count = 0

global all_data_norm
all_data_norm = []

###############################################################################
#mu is the location parameter for the logistic fit of each play type interaction
global mu_array
mu_array = mu_array_n

#s is the scale parameter for the logistic fit of each play type interaction
global s_array
s_array = s_array_n
#incompletion probability for each play type
global incomplete_array
incomplete_array = incomplete_array_n

#turnover probabilitiy for each playtype
global turnover_array
turnover_array = turnover_array_n
#probability each offensive play type takes a certain amount of time
#Note: for pass plays, this is the time for completions
#1st row is RI, 2nd row RO, 3rd is PS, 4th is PD
#The first item in the list represents the probability the play takes 7 sec
#subsequent entries represent the probability it takes 14,21,28,etc. sec
global time_array
time_array = time_array_n
#same structure as the above array, just examines the time breakdown for incomplete passes
global inc_time_array
inc_time_array = inc_time_array_n




################### SIMPLIFICATIONS FOR THE MODEL #############################


#this list describes the range of yardage chunks that can be gained in a single play
global d_list
d_list = list(range(-3,14))



#represents the possibile score differentials
global score_diff_list
score_diff_list = list(range(-17,18))

#identidies the index in d_list for which the value is 0 for follow on use
global zero_index
zero_index = d_list.index(0)

#lets us scale how many chunks the field will be seperated into
#doing 1 yd intervals is difficult for computing power
#we choose 3.33 yd incr. which splits field into 30 sections
global yds_increment
yds_increment = (10/3)

global max_fd_dist
max_fd_dist = int(round(((10/yds_increment) - (min(d_list) * 3)) + 1))

global time_increment
time_increment = 7

global punt_index
punt_index = np.shape(mu_array)[1] + 0

global fg_index
fg_index = np.shape(mu_array)[1] + 1

global kneel_index
kneel_index = np.shape(mu_array)[1] + 2

###############################################################################

################### STATISTICS FOR THE MODEL ##################################

#overall probability a team makes a PAT...scraped from nflFastR since 2015 season
# because of rule change
global PAT_success
PAT_success = 0.9384414

#overall probability a team succeeds for a 2 pt. conv...scraped from nflFastR
#since 2010 season
global two_ptconv_success
two_ptconv_success = 0.4224652

global onside_success
onside_success = .105

#probability of making a field goal from a number of distances:
#<20,20-29,30-39,40-49,50+
#scraped from data since the 2010 season
global field_goal_success_array
field_goal_success_array = [1.000, 1.000,0.9704505,0.9704505,0.9704505,0.9058323,0.9058323,
                            0.9058323,0.8769941, 0.8769941,0.8769941, 0.6248514, 0.6248514,
                            0.6248514, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#alternatively the mean for the time a punt takes is 10.00063 so we could
#just say a FG takes 1 or 2 ticks


#probability a punt travels __ chunks. From 0 --> 30
global punt_chunk_array
punt_chunk_array = [0.0063172204, 0.0004797889,0.0002798769, 0.0006797009, 
                    0.0006397185, 0.0011994722, 0.0031985926, 0.0054775899, 
                    0.0116748631, 0.0367438327, 0.0478989245, 0.0725280876, 
                    0.1341809604, 0.1201471353, 0.1222662029, 0.1574907041, 
                    0.1055935388, 0.0787253608, 0.0634520811, 0.0185518372, 
                    0.0071968334, 0.0038383111, 0.0008796130, 0.0003598417, 
                    0.0001999120, 0,0,0,0,0,0]
#probability a punt takes ___ ticks. From 0 --> 11
#alternatively the mean for the time a punt takes is 13.1158 so we could just
#say it takes 2 ticks

global off_splits
off_splits = []

global def_splits
def_splits = []

#ADD TO THE MODEL WISH LIST
#Field goal, punt, timeouts, two-min warning, 2 min specific strategy

###############################################################################

################### FUNCTIONS #################################################
    
def yardage_distribution_table(yds_increment):
    #this creates a 3 dimmensional array that indicates the probability
    #an offense gets a certain amount of yard chunks according to the offensive & 
    #defensive playcalls
    global mu_array
    mu = mu_array
    global s_array
    s = s_array
    global turnover_array
    global d_list
    d = d_list
    global zero_index
    x_shape = np.shape(mu)[0]
    y_shape = np.shape(mu)[1]
    d_shape = np.shape(d)[0]
    table1 = np.empty((x_shape,y_shape,d_shape))
    for x in range(np.shape(mu)[0]):
        for y in range(np.shape(mu)[1]):
            for d_1 in range(np.shape(d)[0]):
                 table1[x,y,d_1] = (logistic.cdf((yds_increment)*d[d_1]+(yds_increment/2),mu[x,y],s[x,y]) - \
                       logistic.cdf((yds_increment)*d[d_1]-(yds_increment/2),mu[x,y],s[x,y]))
    #the above line rounds yardage gained to the nearest chunk; note 
    #we are using a logistic fit for this data
            normalizing_factor = sum(table1[x,y])
    #ensures each table sums to 1 for probabilities sake
            table1[x,y] = (table1[x,y] / normalizing_factor)
    return table1

def time_distribution_table():
    #this creates a 3 dimmensional array that indicates the probability
    #a play takes a certain amount of time blocks according to the offensive 
    #and defensive playcalls
    global mu_array
    mu = mu_array
    global time_array
    x_shape = np.shape(mu)[0]
    y_shape = np.shape(mu)[1]
    t_shape = np.shape(time_array)[1]
    table1 = np.empty((x_shape,y_shape,t_shape))
    for x in range(np.shape(mu)[0]):
        for y in range(np.shape(mu)[1]):
            for t_1 in range(np.shape(time_array)[1]):
                 table1[x,y,t_1] = time_array[x,t_1]
            normalizing_factor = sum(table1[x,y])
            table1[x,y] = (table1[x,y] / normalizing_factor)
    #ensures each table sums to 1 for probabilities sake
    return table1

def known_table_maker():
    #initializes a storage location for a table that will store the win probabilities
    #for game scenarios that have already been calculated. This will help the
    #model run faster because it avoids the need for recomputing already
    #calculated values
    global known_table
    global max_fd_dist
    global yds_increment
    try:
        known_table
        print("Im using the old one")
    except NameError:
        known_table = -1 * np.ones((4,round(100/yds_increment),130,max_fd_dist,1 + max(score_diff_list)*2))
        print("Im making a new one")
    known_table[:] = known_table
    return known_table
known_table = known_table_maker()

    
def onside_possibilities(d_list, payoff_probabilities,time_left,score_diff):
    global onside_success
    if time <= 0:
        if score_diff < 0:
            winprob = 0
        elif score_diff >0:
            winprob = 1
        else:
            winprob = .5
    else:
        winprob_onside = (dnd_prob_DISTR(d_list, payoff_probabilities, 1, round(55/(yds_increment)), \
                                         round(10/(yds_increment)), time_left, score_diff))* onside_success + \
                      (1 - dnd_prob_DISTR(d_list, payoff_probabilities, 1, round(45/(yds_increment)), \
                                         round(10/(yds_increment)), time_left, -score_diff))* (1-onside_success)
        winprob_kick = (1 - dnd_prob_DISTR(d_list, payoff_probabilities, 1, round(75/(yds_increment)), \
                                         round(10/(yds_increment)), time_left, -score_diff))
        winprob = max(winprob_onside, winprob_kick)
    return winprob


def probz_maker_DISTR(d_list,payoff_probabilities,down,endzone_distance,firstdown_distance,time_left, score_diff):
    #creates a matrix of probabilities (called probz) of winning from a starting down, distance from
    #the endzone, distance from a first down, time left in the game, and score diff.
    global known_table
    global incomplete_array
    global turnover_array  
    global time_array
    global inc_time_array
    global zero_index
    global punt_chunk_array
    global field_goal_success_array
    global punt_chunk_array
    global time_increment
    global punt_index
    global fg_index
    S = payoff_probabilities
    probz = np.zeros((S.shape[0],S.shape[1]+3))
    for x in range(S.shape[0]):
        for y in range(S.shape[1]):
            for d in range(S.shape[2]):
                for t in range(time_array.shape[1]):
    #for all possible offensive, defensive playcall, all the possible yardage
    #gained per play and all the possible plays do the following:
                    
                    if d < endzone_distance:
                        if time_left <= round(120/(time_increment)) or time_left - t > round(120/(time_increment)):
                            probz[x,y] = probz[x,y] + (S[x][y][d] * time_array[x,t] * dnd_prob_DISTR(d_list,S,down+1, (endzone_distance - d - min(d_list)), \
                             (firstdown_distance - d - min(d_list)), time_left-(t+1), score_diff))*(1 - incomplete_array[x][y] - turnover_array[x][y])
                        else:
                            probz[x,y] = probz[x,y] + (S[x][y][d] * time_array[x,t] * dnd_prob_DISTR(d_list,S,down+1, (endzone_distance - d - min(d_list)), \
                             (firstdown_distance - d - min(d_list)), round(120/(time_increment)), score_diff))*(1 - incomplete_array[x][y] - turnover_array[x][y])
                    else:
                        probz[x,y] = probz[x,y] + (S[x][y][d] * time_array[x,t] * dnd_prob_DISTR(d_list,S,down+1, (endzone_distance - d - min(d_list)), \
                             (firstdown_distance - d - min(d_list)), time_left-(1), score_diff))*(1 - incomplete_array[x][y] - turnover_array[x][y])
        
                        
    #probz is prob of winning for the given parameters, and offensive and defensive playcalls, yard chunks gained,
    #and time for a play. S[x][y][d] is the entry in the 3d matrix of probabilities that given
    #x and y playcalls are chosen, you get d yards. time_array[x,t] is an entry in a
    #probability array that an offensive play choice x takes t ticks. dnd_prob 
    #is the next function definded that returns the probability of winning from the resulting
    #position due to the previous play's result. We multiply this entire fxn
    #by the probability of completing a pass (because S is payoff for completed passes only)
    #This is the root of the mathematics for our model
    #1ST ELSE stops the clock in the event of a two min warning
    #2ND ELSE statement makes a touchdown scoring play last only one time tick

                    if t == 0:
                        probz[x,y] = probz[x,y] + (S[x][y][d] * 1 *(1-dnd_prob_DISTR(d_list, payoff_probabilities, 1, round(100/yds_increment) - \
                             endzone_distance, round(10/(yds_increment)), time_left-1, -score_diff)))*(turnover_array[x][y])     
    #this is similar to the above equation. It instead has the parameters changed to
    #match the resulting parameters for a turnover
    # t = 0 is used because we have assumed each turnover play will take only
    #one time tick.
                    if d == zero_index:
                        probz[x,y] = probz[x,y] + inc_time_array[x,t] * dnd_prob_DISTR(d_list,S,down+1, (endzone_distance - d - min(d_list)), \
                             (firstdown_distance - d - min(d_list)), time_left-(t+1), score_diff)*(incomplete_array[x][y])
    #this is also similar to the top equation. It has the paraemeters changed to 
    #match an incomplete pass. d = zero.index for this scenario because
    #all incomplete passes result in 0 yds gained
        for d in range(0,30):  
    #PUNT LOOP. Can revisit later to see if we want to change punt distances to a predetermined value
            probz[x,punt_index] = probz[x,punt_index] + (punt_chunk_array[d] * 1 * (1 - dnd_prob_DISTR(d_list,S, 1, round(100/yds_increment) - (endzone_distance - d - 1), \
                             round(10/yds_increment), time_left-(2), -score_diff)))
    #FG LOOP. SUCCESS OR MISS?    
        probz[x,fg_index] = probz[x,fg_index] + (1 * 1 * onside_possibilities(d_list, payoff_probabilities, time_left-1, score_diff + 3) * \
                                     (field_goal_success_array[endzone_distance]))
        probz[x,fg_index] = probz[x,fg_index] + (1 * 1 * (1 - dnd_prob_DISTR(d_list, payoff_probabilities, 1, round(100/yds_increment) - endzone_distance - 2, \
                                     round(10/(yds_increment)), time_left-(1), -score_diff))) * (1 - field_goal_success_array[endzone_distance])
        probz[x,kneel_index] = probz[x,kneel_index] + (1 * 1 * (dnd_prob_DISTR(d_list, payoff_probabilities, down + 1, endzone_distance + 1, \
                                     firstdown_distance + 1, time_left-(round(45/time_increment)), score_diff)))

    return probz

def dnd_prob_DISTR(d_list,payoff_probabilities,down,endzone_distance,firstdown_distance,time_left, score_diff):
    #this function and the probz_maker function continually call one another...
    #this represents the recursive nature of our model
    #THE BIG EVENT CHECKER:
    global known_table
    global yds_increment
    global score_diff_list
    global time_array
    global PAT_success
    global two_ptconv_success
    global punt_chunk_array
    global field_goal_success_array
    global splits
    global play_count
    global max_fd_dist
    #A team scores; how many points do they get?
    if endzone_distance <= 0:
        winprob_PAT = onside_possibilities(d_list, payoff_probabilities, time_left, score_diff + 7) * PAT_success + \
                  onside_possibilities(d_list, payoff_probabilities, time_left, score_diff + 6) * (1-PAT_success)
        winprob_twopt = onside_possibilities(d_list, payoff_probabilities, time_left, score_diff + 8)* two_ptconv_success + \
                  onside_possibilities(d_list, payoff_probabilities, time_left, score_diff + 6)* (1-two_ptconv_success)
        winprob = max(winprob_PAT, winprob_twopt)
    #Time runs out. Who wins?
    elif time_left <= 0:
        if score_diff > 0:
            winprob = 1
        elif score_diff < 0:
            winprob = 0
        else:
            winprob = .5
    #The winning team has the ball with 4 or less time chunks left = they win
    elif score_diff > 0 and time_left <= (round(45/time_increment) * (4-down)):
        winprob = 1
    #Bring ball out to the 20 if a punt goes into the endzone: Touchback condition
    #for punts    
    elif down == 1 and endzone_distance >= round(100/(yds_increment)):    
        winprob = dnd_prob_DISTR(d_list, payoff_probabilities, 1, round(80/(yds_increment)), \
                                     round(10/(yds_increment)), time_left, score_diff)
    #If the distance to endzone > 100 yards = Safety
    elif endzone_distance >= round(100/(yds_increment)):
        winprob = 1 - dnd_prob_DISTR(d_list, payoff_probabilities, 1, round(75/(yds_increment)), \
                                     round(10/(yds_increment)), time_left, -score_diff+2)
    #If the score differential exceeds the cap we currently have as the max, we say the team
    #in the lead wins
    elif score_diff >= max(score_diff_list):
        winprob = 1
    #Same as above, but the opposite. This one is a losing condition
    elif score_diff <= min(score_diff_list):
        winprob = 0
    #First down converted condition
    elif firstdown_distance <= 0:
        winprob = dnd_prob_DISTR(d_list, payoff_probabilities, 1, endzone_distance,round(10/(yds_increment)), time_left, score_diff)
    #Turnover on downs condition
    elif down >= 5:
        winprob = 1 - dnd_prob_DISTR(d_list, payoff_probabilities, 1, round(100/yds_increment) - endzone_distance, \
                                     round(10/(yds_increment)), time_left, -score_diff)
    #The "_____ and goal" condition. Does not allow for a first down for the remainder
    #of the drive
    elif firstdown_distance >= endzone_distance and firstdown_distance != max_fd_dist:
        winprob = dnd_prob_DISTR(d_list, payoff_probabilities, down, endzone_distance, max_fd_dist, time_left, score_diff)        
    #If the known_table value is known, just grab that value instead of calculating it
    elif known_table[down-1, endzone_distance-1, time_left-1, firstdown_distance-1, score_diff - min(score_diff_list)] != -1:
        winprob = known_table[down-1,endzone_distance-1,time_left-1, firstdown_distance-1, score_diff - min(score_diff_list)]
        return winprob
    else:
    #If none of the above conditions are met, caluclate prob of winning using nash.py and probz_maker
    #We use nash due to the game theoretic approach of our model. This nash.py function calculates
    #the optimal mixed strategy for both the offense and the defense
#        print(down, endzone_distance, time_left, firstdown_distance, score_diff)
        probz = probz_maker_DISTR(d_list,payoff_probabilities,down,endzone_distance,firstdown_distance, time_left, score_diff)
#        print(probz)
        solution = nash_equil(probz)
        winprob = solution[0]
        off_strat = solution[1]
        def_strat = solution[2]
        all_data_norm.append((down, endzone_distance, time_left, firstdown_distance, score_diff, winprob, off_strat, def_strat))
        if max(off_strat) < .999:
            # print(down, endzone_distance, time_left, firstdown_distance, score_diff)
            off_splits.append((off_strat, down, endzone_distance, time_left, firstdown_distance, score_diff))
        if max(def_strat) < .999:
            # print(down, endzone_distance, time_left, firstdown_distance, score_diff)
            def_splits.append((def_strat, down, endzone_distance, time_left, firstdown_distance, score_diff))
        
        play_count += 1
        if play_count%500 == 0:
            print(play_count)

        
        known_table[down-1,endzone_distance-1,time_left-1, firstdown_distance-1,score_diff-min(score_diff_list)] = winprob
        #this dot product is the product of the optimal offensive mixed strategy
        #and the probz matrix. This product is then dotted with the oprimal 
        #defensive mixed strategy
    known_table = known_table
    return winprob

###############################################################################

################### RUNNING THE MODEL #########################################


ydg_dist_tbl = (yardage_distribution_table(yds_increment))

down = 1
endzone_distance = 15
time = 130
firstdown_distance = 3
score_diff = 0


start = tme.time()
print(dnd_prob_DISTR(d_list,ydg_dist_tbl,down,endzone_distance,firstdown_distance,time,score_diff))
end = tme.time()
print((str((end - start)/60)) + " minutes")

#np.unravel_index(np.argmax(known_table), np.shape(known_table))

#DOWN = 0TH INDEX, WHAT DOWN IS IT?
#ENDZONE DISTANCE = 1ST INDEX, HOW MANY DISTANCE BLOCKS UNTIL THE ENDZONE?
#TIME = 2ND INDEX, HOW MANY TIME BLOCKS LEFT IN THE GAME?
#FIRST DOWN DISTANCE = 3RD INDEX, HOW MANY DISTANCE BLOCKS UNTIL A 1ST DOWN CONVERSION?
#SCORE DIFF = 4TH INDEX, HOW MUCH IS THE TEAM THAT HAS THE BALL LEADING BY?
