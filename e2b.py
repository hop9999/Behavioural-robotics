import gym
import numpy as np
import time

env = gym.make('CartPole-v0')

pvariance = 0.1
ppvariance = 0.02
nhiddens = 5
ninputs = env.observation_space.shape[0]

if (isinstance(env.action_space, gym.spaces.box.Box)):
 noutputs = env.action_space.shape[0]
else:
 noutputs = env.action_space.n

b1 = np.zeros(shape=(nhiddens, 1)) # bias first layer
b2 = np.zeros(shape=(noutputs, 1)) # bias second layer
W1_ar = []
W2_ar = []

population_size = 10
population_iterations = 10
simulation_time = 200

for j in range(population_size):
    W1 = np.random.randn(nhiddens,ninputs) * pvariance # first layer
    W2 = np.random.randn(noutputs, nhiddens) * pvariance # second layer
    W1_ar.append(W1)
    W2_ar.append(W2)


for _ in range(population_iterations):
    score_ar = []
    for k in range(population_size):
        env.reset()
        score = 0
        for i in range(simulation_time):
            env.render()
            if i == 0:
                observation, reward, done, info = env.step(env.action_space.sample())
            else:
                observation.resize(ninputs,1)
                Z1 = np.dot(W1_ar[k], observation) + b1
                A1 = np.tanh(Z1)
                Z2 = np.dot(W2_ar[k], A1) + b2
                A2 = np.tanh(Z2)
                if (isinstance(env.action_space, gym.spaces.box.Box)):
                    action = A2
                else:
                    action = np.argmax(A2)
                observation, reward, done, info = env.step(action)
            score += reward
        score_ar.append(score)
        env.close()

    print(score_ar)
    W1_ar_new = []
    W2_ar_new = []
    for k in range(population_size):
        if k < population_size/2:
            ind = score_ar.index(max(score_ar))
            W1_ar_new.append(W1_ar[ind])
            W2_ar_new.append(W2_ar[ind])
            score_ar[ind] = 0
        else:
            W1_ar_new.append(W1_ar_new[k - int(population_size/2)] + np.random.randn(nhiddens,ninputs)*ppvariance)
            W2_ar_new.append(W2_ar_new[k - int(population_size/2)] + np.random.randn(noutputs, nhiddens)*ppvariance)
    W1_ar = W1_ar_new
    W2_ar = W2_ar_new
    
