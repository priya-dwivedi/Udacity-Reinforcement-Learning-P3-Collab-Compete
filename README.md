## Udacity Reinforcement Learning Nanodegree P3 - Collab Compete
This repo has the code for training two agents to play a game of ping pong.There are 3 main files in this submission:
* Tennis.ipynb - This has the code to start the environment, train the agent and then test the trained agent
* maddpg_agent.py - This file has the implementation of the DDPG Agent which is used by Tennis.ipynb 
* model.py - This file has the neural network that is trained to do the funcion approximation

## About the project environment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. The value for each action should be between -1 and +1. 

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. The environment is considered solved, when the average (over 100 episodes) of the max score is at least +0.5.

## Setting up the Python Enviroment
The following libraries are needed to run the code:
1. unityagents - ```pip install unityagents```
To see the agent in action, please download the unity environment to your local
Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
2. Separately download the headless version of the environment. I found that the training proceeded faster with this option.[this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
3. Pytorch - Install latest version based on your OS and machine from https://pytorch.org/get-started/locally/
4. Numpy, Matplotlib


## Training and Visualizing the trained agent
The code for training the agent is in the notebook Tennis.ipynb.

### Training an agent

For training the agent, I chose the headless agent Linux envtt. You can load the envtt as below:
```
env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

Section 1.1 of the notebook has the code for training an agent. The command below sets up the two DDPG agents. 
```
from maddpg_agent import Agent
agent_0 = Agent(state_size, action_size, random_seed=0)
agent_1 = Agent(state_size, action_size, random_seed=0)
```
The function maddpg() resets the environment, provides episodes to train the agent, gets actions from agent class and passes the results of the actions (next state, reward) to the agent class.

To train the model. Set train to True and run below:
```
train = True
if train:
    scores, moving_average = maddpg()
```

### Visualizing the trained agent
Section 1.2 of the notebook has the code for playing in the Unity Environment with the trained agent. The main steps are:
1. Initialize the Envtt with visualization
2. Load the trained weights into the agent
```
agent_0 = Agent(state_size, action_size, random_seed=0)
agent_1 = Agent(state_size, action_size, random_seed=0)
agent_0.actor_local.load_state_dict(torch.load('checkpoint_actor_0.pth'))
agent_0.critic_local.load_state_dict(torch.load('checkpoint_critic_0.pth'))
agent_1.actor_local.load_state_dict(torch.load('checkpoint_actor_1.pth'))
agent_1.critic_local.load_state_dict(torch.load('checkpoint_critic_1.pth'))
```

2. Play a game

```
scores = []
for i in range(1, 6):                                      # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores_all_agents = np.zeros(num_agents)               # initialize the score (for each agent)
    while True:
        action_0 = agent_0.act(states[0], add_noise = False)    
        action_1 = agent_1.act(states[1], add_noise = False)
        actions = np.concatenate((action_0, action_1), axis=0).flatten()
        env_info = env.step(actions)[brain_name]           
        next_states = env_info.vector_observations         
        rewards = env_info.rewards                         
        dones = env_info.local_done                        
        scores_all_agents += env_info.rewards                        
        states = next_states                              
        if np.any(dones):                                 
            break
    scores.append(np.max(scores_all_agents))
print("Scores are: ", scores)

```



