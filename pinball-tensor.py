""" Majority of this code was copied directly from Andrej Karpathy's gist:
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5 """

""" Trains an agent with (stochastic) Policy Gradients on Pinball. Uses OpenAI Gym. """
import numpy as np
import _pickle as pickle
import gym
import time
import sys
import random

from gym import wrappers

# hyperparameters to tune
H = 200 # number of hidden layer neurons
A = 6 # number of actions
batch_size = 5 # used to perform a RMS prop param update every batch_size steps
learning_rate = 1e-3 # learning rate used in RMS prop
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

# Config flags - video output and res
resume = False # resume training from previous checkpoint (from save.p  file)?
render = True # render video output?

# model initialization
xvalues = [[],[],[],[],[],[]]
D = 94*80 # input dimensionality: 94x80 grid
if resume:
  model = pickle.load(open('save-pinball.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H, D) / np.sqrt(D) # "Xavier" initialization - Shape will be H x D
  model['W2'] = np.random.randn(A, H) / np.sqrt(H) # Shape will be A x H

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 250x160x3 uint8 frame into 7,520 (94x80) 1D float vector """
  I = I[29:217] # crop - remove 29px from start & 33px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
  I = I[::2,::2,0] # downsample by factor of 2, only 1 value
  I[I != 0] = 1 #greyscale
  return I.astype(float).ravel() # ravel flattens an array and collapses it into a column vector


env = gym.make("VideoPinball-v0")
#env = wrappers.Monitor(env, 'tmp/pong-base', force=True) # record the game as as an mp4 file
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
actionDistribution = [0,0,0,0,0,0]

reward_benchmark = 100000
reward_ratio = 10000

while True:
  if render: env.render()
  
  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  # we take the difference in the pixel input, since this is more likely to account for interesting information
  # e.g. motion
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)

  # The following step is randomly choosing a number which is the basis of making an action decision
  # If the random number is less than the probability of UP output from our neural network given the image
  # then go down.  The randomness introduces 'exploration' of the Agent
  #action = 2 if np.random.uniform() < aprob else 3 # roll the dice! 2 is UP, 3 is DOWN, 0 is stay the same
  #2 is both paddles up, 3 is right paddle up, 4 is left paddle up, 5 is pull bumper back, 6 fire bumper, except maybe 1 is?
  maxProb = max(aprob)
  maxIndex = aprob.index(maxProb)
  action = maxIndex
  rand = random.random()
  r = [i for i in range(A) if i != maxIndex]
  if rand > maxProb:
    action = random.choices(r)[0]
  actionDistribution[action] += 1
  
  # record various intermediates (needed later for backprop).
  # This code would have otherwise been handled by a NN library
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = []
  for i in range(0,A):
    y.append(1 if (maxIndex == i and action == i) else 0) # a "fake label" - this is the label that we're passing to the neural network
  # to fake labels for supervised learning. It's fake because it is generated algorithmically, and not based
  # on a ground truth, as is typically the case for Supervised learning

  dlogps.append(y - aprob[action]) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
  #print(len(dlogps))
  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1
    sub = reward_benchmark/len(drs)
    drs[:] = [(e - sub)/reward_ratio for e in drs]
    reward_sum = sum(drs)
    actionDistribution = [0,0,0,0,0,0]
    avgs = []
    for i in range(0,A):
      avgs.append(sum(xvalues[i]) / len(xvalues[i]))
    #print(avgs)
    xvalues = [[],[],[],[],[],[]]
    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    if (np.std(discounted_epr) != 0):
      discounted_epr /= np.std(discounted_epr)
    epdlogp *= discounted_epr # modulate the gradient with advantage (Policy Grad magic happens right here).
    grad = policy_backward(eph, epx, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch
    
    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print ('resetting env. episode #' + str(episode_number) + ' reward total was %f. running mean: %f' % (reward_sum, running_reward))
    if episode_number % 100 == 0: pickle.dump(model, open('save-pinball.p', 'wb'))
    if running_reward > -0.5: reward_benchmark *= 2
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None
