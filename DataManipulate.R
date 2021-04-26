# BBQ Data
# Customers
# Variables  
#  Time - The arrival time of the customer
#  Group - Group ID number for each customer 
#  Entity.ID - Entity number in corresponding group

# Reading in data, necessary libraries
library(readr)
library(dplyr)
pbp = read_csv('play_by_play_2001.csv')

#makes 2 new data sets, one with only run plays, the other with only pass plays
pbp_runs <- pbp %>%
  filter(play_type == 'run')
pbp_pass <- pbp %>%
  filter(play_type == 'pass')

#grabs yards gained for all plays of given play type
RunYardsPerPlay <- pbp_runs$yards_gained
PassYardsPerPlay <- pbp_pass$yards_gained


#makes histograms of yards gained for each type of play
hist(RunYardsPerPlay, xlab="Yards Gained on Individual Run Plays", 
     main = "Histogram of Yardage Gained on Run Plays", breaks = "FD", xlim = c(0,100),ylim = c(0,1750))
hist(PassYardsPerPlay, xlab="Yards Gained on Individual Pass Plays", 
     main = "Histogram of Yardage Gained on Pass Plays", breaks = "FD", xlim = c(0,100),ylim = c(0,1000))
            #amount of incomplete passes (ie yards gained = 0) is in the neighborhood of 7000
#looks to me like a normal distribution. I think we can fit it

pbp_runsleft <- pbp %>%
  filter(play_type == 'run' & run_location == "left")
RunYardsPerPlayLeft <- pbp_runsleft$yards_gained
hist(RunYardsPerPlayLeft, xlab="Yards Gained on Individual Left Run", 
     main = "Histogram of Yardage Gained on Left Run Plays", breaks = "FD", xlim = c(0,100),ylim = c(0,600))


pbp_runsmiddle <- pbp %>%
  filter(play_type == 'run' & run_location == "middle")
RunYardsPerPlayMiddle <- pbp_runsmiddle$yards_gained
hist(RunYardsPerPlayMiddle, xlab="Yards Gained on Individual Middle Run", 
     main = "Histogram of Yardage Gained on Middle Run Plays", breaks = "FD", xlim = c(0,100),ylim = c(0,600))

pbp_runsright <- pbp %>%
  filter(play_type == 'run' & run_location == "right")
RunYardsPerPlayRight <- pbp_runsright$yards_gained
hist(RunYardsPerPlayRight, xlab="Yards Gained on Individual Right Run", 
     main = "Histogram of Yardage Gained on Right Run Plays", breaks = "FD", xlim = c(0,100),ylim = c(0,700))

pbp_runsguard <- pbp %>%
  filter(play_type == 'run' & run_gap == "guard")
RunYardsPerPlayGuard <- pbp_runsguard$yards_gained
hist(RunYardsPerPlayGuard, xlab="Yards Gained on Individual Guard Runs", 
     main = "Histogram of Yardage Gained on Guard Run Plays", breaks = "FD", xlim = c(0,100),ylim = c(0,500))

pbp_runstackle <- pbp %>%
  filter(play_type == 'run' & run_gap == "tackle")
RunYardsPerPlayTackle <- pbp_runstackle$yards_gained
hist(RunYardsPerPlayTackle, xlab="Yards Gained on Individual Tackle Run", 
     main = "Histogram of Yardage Gained on Tackle Run Plays", breaks = "FD", xlim = c(0,100),ylim = c(0,500))

pbp_runsend <- pbp %>%
  filter(play_type == 'run' & run_gap == "end")
RunYardsPerPlayEnd <- pbp_runsend$yards_gained
hist(RunYardsPerPlayEnd, xlab="Yards Gained on Individual End Run", 
     main = "Histogram of Yardage Gained on End Run Plays", breaks = "FD", xlim = c(0,100),ylim = c(0,400))

# random.discrete to determine group size (ex. percent 1 == 0.3449 -> 34.49% chance that group size is 1)


##########################################################################################################

# Arrivals # vector of groups interarrival times
BBQData.Times <- BBQData.Customers$Time[BBQData.Customers$GroupNumberID==1]

tab.arrival = table(cut(BBQData.Times,'day')) # arrivals/day -> separate days, separate day from arrivals
as.data.frame(tab.arrival) 


###########
# We create vectors that mark the hour of each day and the daily arrivals that occur during that hour. Recall, there are 24 hours in a day (denoted by hour 0, hour 1, ..., hour 23) and 91 days in the timeperiod of the data. 
Hours <- rep(c(0:23), 27)

DailyArrivals <- rep(rep(0, 23), 27)

StartTime = as.POSIXct("04/05/2020 10:00",format = "%m/%d/%Y %H:%M")


#Loop through each day 
for(j in 1:27)
{
  #Loop through each hour of the day
  for(i in 1:24)
  {
    #We find all the arrivals that occur between hours i-1 and i on day j
    DailyArrivals[(j-1)*24+i] <- length(which(BBQData.Times < StartTime+24*60*60*(j-1) +
                                                60*60*i & BBQData.Times > StartTime+24*60*60*(j-1)+
                                                60*60*(i-1)))
  }
}


#Plot the Daily Arrivals according to the hour of theday
plot(Hours, DailyArrivals)

MeanHours <- c(0:23)
MeanDailyArrivals <- rep(0, 24)
for(i in 1:24)
{
  MeanDailyArrivals[i] = mean(DailyArrivals[Hours == i-1])
}

MeanDailyArrivals

plot(MeanHours, MeanDailyArrivals, ylim = c(0, 20))





length(BBQData.Times)


InterarrivalTimes <- difftime(BBQData.Times[2:11762],
                              BBQData.Times[1:11761], units =  "secs")
InterarrivalTimes <- unclass(InterarrivalTimes)
InterarrivalTimes
InterarrivalTimes <- InterarrivalTimes[1:11762]
InterarrivalTimes

hist(InterarrivalTimes, xlab="Interarrival Times (seconds)", 
     main = "Histogram of Interarrival Times", breaks = "FD", xlim = c(0,1000),ylim = c(0,1500))



























#distribution fitting

library(fitdistrplus)

InterarrivalExpoFit <- fitdist(InterarrivalTimes, "exp" )
summary(InterarrivalExpoFit)

InterarrivalGammaFit <- fitdist(InterarrivalTimes, "gamma")
summary(InterarrivalGammaFit)

Normalfit = fitdist(InterarrivalTimes, 'norm')
summary(Normalfit)

#gof stats/ks tests

gofstat(InterarrivalExpoFit)
gofstat(InterarrivalExpoFit)$kstest

gofstat(InterarrivalGammaFit)
gofstat(InterarrivalGammaFit)$kstest

gofstat(Normalfit)
gofstat(Normalfit)$kstest
