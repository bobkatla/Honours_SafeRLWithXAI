# Explainable Safe Reinforcement Learning
---
## Description
This project is a research project on how to improved the explain-ability of safety behaviours of the agent during the exploration of Reinfocement Learning (RL). The idea of the project is to have an explanation model that is a Neural Network (NN) resulted from Deep Learning (DL) that can help the agent know what are the potential hazard and give advice to the agent so the agent can avoid the hazard and explain its own safety action. We were also interest in how different data budget for inputting the DL can affect the performance.

At the current state, the project is at the state that can create the DL model and run the DL with RL to and view how will that affect the hazard encounter and the reward of the agent. We have done the test 2 environments and test with 2 classic RL methods: Q-learning and SARSA. Further work can be done on updating the explanation model with new data during exploration and ebtablishing a teacher-student module to help the model getting better.

## Extra resources
The work log of me working with the supervisor can be found here, and the source code of my thesis (Latex) can be found here. 

## How to run it
To run or re-build the project, we can do it by understanding the components in this repository. In general, there are some main folders to consider. The first would be the ./code/runcode/storeCodeWorkoutlater as it would be where I create all the main files and run the general code (testing the Q-learning and SARSA without the DL). In this folder, there would be 2 folders: w1 and w2. These two are almost identical with only different in the params files to create the world. The second folder is the ./code/runcode/result_testing/remote as it the file to run DL and DL with RL. In here, we can find 2 folders for 2 worlds the same as above. In eachh world we will find other folders for each test cases of different data input of DL.

### Creating the world
Navigating to any of the folder above, we can find the group of files: params.py, object_world.py and hazard.py. These 3 files will help create the grid world for the navigation task. To create or change the world, we only have to make change to the params.py file. To see how the world look like, just run:

`python params.py`

This should show the current world.

### Run the normal Q-learning and SARSA
Navigating to one of the general space, such as the world 1:

`cd code/runcode/storeCodeWorkoutlater/w1`

We can then run the normal Q-learning without DL by running:

`python hazardCountNormal.py`

This would output the hazard counting in form of picture with the ha_count_normal_RL.png and in .npy with Q_ha.npy; the same thing with the reward for reward_normal_RL.png and Q_re.py. 

We will have the similar output with SARSA by running:

`python SARSAnormal.py`

### Creating data
In the same general folder as above, we can create a list of randome data for testing or work as input for DL. You need to open the file testWorldObject.py, and fix the code for outputing the name and form (default .csv) of the file and the number of data you want to create. Then you save and run: 

`python testWorldObject.py`

The file can finish really fast and early and it will in form you that to "Break now", when that happens just `ctrl + C` to end the running and the needed file will exist in the same folder.

### Running the DL
Navigating to any of the test case in the remote folder such as:

`cd code/runcode/result_testing/remote/test1_100`

In here we can find the group of files for DL: newDL.py, data_preparation.py and a .csv file. The .csv file would be the input data for training, open the data_preparation.py to make sure that the name match. You can then start to run the training of DL by: 

`python newDL.py`

As the results, it would output model_weights.h5 and model_architecture.json for the saving the NN and the scaler.pkl for saving the scaler. You would need all 3 files for running the model-loading.py file later for hazard prediction. 

### Running the integration of DL and RL
Still in the same above folder, you can find betterHazardCount.py for Q-learning integration and betterSARSA.py for SARSA integration. All you have to do is make sure that the 3 models files and the model-loading.py are in the same place and then run the file such as:

`python betterSARSA.py`

The results would be similar to the running of normal no DL with the images and .npy files. This would help for comparing the resuls later. 
