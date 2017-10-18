'''

little james got addicted to ML, especially the kind which he can't handle. so here i come, sprinkle some computer science magic, in the hope of elimitating all the bugs and providing a smooth experience.

python three(3) should be used to run this code. here we're on python3.5
if you have Anaconda installed, you should be using py3.5 when you are not in virtual envs.

to check, simply enter:

python --version

call 911 in case of emergency.

'''

'''

pip install h5py
git clone https://github.com/matthiasplappert/keras-rl

# because we want the code of keras-rl accessible locally, and be able to update immediatly on change. if you pip install, it would be harder to achieve the same.

cd keras-rl
pip install -e .

# install from current directory. -e means install in dev mode.

python examples/dqn_cartpole.py

# we should now see cartpole swinging. this is to check if everything's right.


now let's copy the gosh darn NAF example here and make some necessary changes.
'''

import numpy as np
import gym

'''

note: little james does not seem to know the gym API. remember to ask him, and teach him about it if he really doesn't. kinda important.

'''

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor
from rl.keras_future import concatenate, Model

from rl.callbacks import Visualizer

class PendulumProcessor(Processor):
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward / 100.


'''
note: change the name to bipedal.

'''

ENV_NAME = 'Pendulum-v0'
ENV_NAME = 'BipedalWalker-v2'
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Build all necessary models: V, mu, and L networks.
'''

this network is trained to predict V(value function) given state.

'''
V_model = Sequential()
V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(1))
V_model.add(Activation('linear'))
print(V_model.summary())


'''

this network is trained to predict mu(action) given state.
you may call it the "actor". this is the 'policy' you are trying to improve.
after enough training, it should generate optimal(max future reward) actions for any given state. it's your baby. it's the core of Bob.

'''
mu_model = Sequential()
mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(nb_actions))
mu_model.add(Activation('linear'))
print(mu_model.summary())

'''

the following network is trained to predict slanted "L", which predicts a "lower triangle matrix". call it L-network.

the author used some tricks: instead of using a "critic", or "Q-network" to predict Q(s,a), he uses "V-network" above to predict V(s), then estimate A(s,a) with some TECHNIQUE, then let Q(s,a) = V(s) + A(s,a). clever.

ok, how did he estimate A(s,a)? what TECHNIQUE?

per his paper,

A(s,a) = -1/2 (a-mu(s)).transpose().matmul(P(s)).matmul((a-mu(s)))

where P(s) = L(s).matmul(L(s).transpose()).

where L(s) is the output of L-network, given state s.

i know, it's full of linear algebra. i don't have simpler terms for this except linear algebra. sorry.

in simple words, we can train L-network to give us the perfect L-matrix, that minimizes the error when estimating A(s,a) with it.

since the prediction of Q is now separated into the prediction of V and A, the accuracy of prediction increased, and the agent should learn better than with just a Q network(as in DDPG).

which is the only difference between NAF, and vanilla DDPG.

'''

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
x = concatenate([action_input, Flatten()(observation_input)])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)

'''

you will realize that the size of this output layer below, is nb_action*(nb_action+1)/2, which makes a good trianglular(half) matrix.

'''

x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
x = Activation('linear')(x)
L_model = Model(input=[action_input, observation_input], output=x)
print(L_model.summary())

'''

the actual tedious computation of V, A, L, Q, from replay memory(experience), and gradient descent and shit, are hidden inside keras-rl. now close your eyes and wish it works.

'''

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
processor = PendulumProcessor()
memory = SequentialMemory(limit=200000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=100, random_process=random_process,
                 gamma=.99, target_model_update=1e-3, processor=processor)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mse'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

'''

bipedal walker env, when submitting to gym network, uses an episodic length of 1600 steps.

disable visualization to accelerate.

to successfully train bipedal, about 4000 episode is needed. that's about 4,000,000 steps. let's use 1M as a starting point!

during the entire duration of fit(), the noise amplitude will gradually decrease from 1x to 0x. this is desired behavior. you will see code doing that within keras-rl.

'''

'''
now let's talk about visualization. by default, if you choose visualize=True, keras-rl will add the following callback, to visualize after every step:

class Visualizer(Callback):
    def on_action_end(self, action, logs):
        self.env.render(mode='human')

as per https://github.com/matthiasplappert/keras-rl/blob/master/rl/callbacks.py#L326-L328

but really, you want controllable visualization rates, at least for bipedal. you can make a similar callback yourself:
'''

class SkippyVisualizer(Visualizer):
    def on_action_end(self, action, logs):
        if not hasattr(self,'counter'):
            self.counter = 0

        self.counter+=1

        if self.counter%20==0: # render() every 20 steps. pretty enough for visualization.
            self.env.render(mode='human')
sv = SkippyVisualizer()

agent.fit(env, nb_steps=1e6, callbacks=[sv], visualize=False, verbose=1, nb_max_episode_steps=1600)

# After training is done, we save the final weights.
# agent.save_weights('cdqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=1600)

'''
think again: is bipedal really 30-fps environment? if not then what fps is it?
'''
