import numpy as np
import matplotlib.pyplot as plt
from hazard import Hazard


############
#world params
a=10
b=10
#######################
world=np.zeros((a,b))
# wall is blue
world[:,0]=1
world[0,:]=1
world[:,b-1]=1
world[a-1,:]=1
##########################
#original environment 1
#horizontals:

# Adding some more hazards
# fire - green
world[3,2]=2
world[3,8]=2
# water - yellow
world[7,2]=3
world[1,6]=3

world[5,1]=1
world[5,9]=1
world[1,5]=1
world[9,5]=1
world[3:8,5]=1
world[5,3:8]=1
#world[9,2]=3#end
'''
#######Environment 2######
world[8,2:10]=1
world[5,1:5]=1
world[5,6:10]=1
#world[4,0:9]=1
world[2,2:6]=1
world[4:5,4]=1
world[2:5,6]=1
#world[9,4]=6
#world[9,9]=3#end
#world[1,9]=3#start
'''
'''
#########Environment 3##########
world[8,2:4]=1
world[8,5:10]=1
world[5,1:5]=1
world[3,2:4]=1
world[5,6:10]=1
#world[4,0:9]=1
world[2,6:8]=1
world[3:5,4]=1
world[2:7,6]=1
world[2,2]=1
#world[4,3]=3#end
'''

##################
############################
targ=np.array([8.,8.])#target location
thresh=0.1#distance threshold
################
NT=np.array([1.,1.])
#Q learning params
Nruns=10#number of runs
alpha=0.05#learning rate
gamma=0.95#discound factor
epsilon=1#exploration parameter
episodes=1000#no. of episodes
A=4#no. of actions
highreward=1#reward for reaching good state
penalty=-0.01#reward for bumping into obstacle

firePelnaty = -0.05
waterPelnaty = -0.03

# NTreward=0
livingpenalty=-0.001 #living reward, if any
breakthresh=200 #max number of interactions per episode
# evalruns=1#no. of evaluation runs for calcret function
# evalsteps=100#no. of evaluation steps
epsilon_decay=0.0005
#newtarg=np.array([8,8])

# plt.imshow(np.rot90(world))
# plt.show()