import os
import random
from collections import deque
import tensorflow as tf
from keras import backend as bk
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import huber_loss
import numpy as np
import copy

import gym
from gym import spaces
from gym.utils import seeding

import fem
import pickle


class PoissonSolverEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(PoissonSolverEnv, self).__init__()
		self.nelems = 5
		self.startp = 3
		self.state = np.ones(self.nelems, dtype=int)*self.startp
		self.maxsteps = 3
		self.dom = [0,1]
		self.lbc = 0.0
		self.rbc = 0.0
		self.maxp = 7
		self.mesh = fem.makeHMesh(self.nelems + 1, 0.0, self.dom, self.state)
		[self.evalB, self.exactSol] = fem.makeSystem(self.dom, self.lbc, self.rbc)


	def step(self, action):
		done = ((np.all(action == 0) and self.current_step > 5)  or self.current_step >= self.maxsteps)
		self.rew = np.zeros(self.nelems)

		if np.sum(action) != 0:
			raise ValueError('Sum of actions does not equal zero.')

		if np.any(self.state + action > self.maxp) or np.any(self.state + action < 1):
			raise ValueError('Max/min polynomial order exceeded. {0}'.format(self.state))

		sig = lambda x: 1./(1 + np.exp(-x))
		oldstate = copy.copy(self.state)
		for i in range(self.nelems):
			self.state = copy.copy(oldstate)
			if self.state[i] < self.maxp - 1:
				self.state[i] += 1
			self.evalSolution()
			err = self.evalError()
			self.rew[i] = sig(-(err - self.err))


		self.state = oldstate + action
		self.evalSolution()
		self.err = self.evalError()

		self.current_step += 1

		return [self.state, self.rew, done, {}]



	def reset(self):
		self.current_step = 0
		self.state = np.ones(self.nelems, dtype=int)*self.startp
		self.evalSolution()
		self.starterr = self.err = self.evalError()
		return self.state

	def render(self, mode='human'):
		return

	def close(self):
		return
	
	def evalSolution(self):
		self.mesh = fem.makeHMesh(self.nelems + 1, 0.0, self.dom, self.state)
		[S, B] = fem.makeGlobalMatrices(self.evalB, self.mesh, self.state, self.lbc, self.rbc)
		self.sol = np.linalg.solve(S,B)

	def evalError(self):
		return fem.getTotalError(self.mesh, self.sol, self.state, 20, self.exactSol)





class DQN():
	def __init__(self,env,options):
		self.env = env
		self.options = options

		self.state_size = np.shape(self.env.state)[0]
		self.action_size = np.shape(self.env.state)[0]

		self.model = self._build_model()
		self.target_model = self._build_model()
		self.policy = self.make_epsilon_greedy_policy()		
		self.nsteps = 0
		self.repbuf = []
		self.repstates = []
		self.repqs = []

	def update_options(self, options):
		self.options = options

	def _build_model(self):
		layers = self.options.layers

		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(layers[0], input_dim=self.state_size, activation='relu'))
		#for l in layers:
		#	model.add(Dense(l, activation='relu'))

		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss=huber_loss,
					  optimizer=Adam(lr=self.options.alpha))

		return model

		




	def update_target_model(self):
		# Copy weights from model to target_model
		self.target_model.set_weights(self.model.get_weights())




	def train_episode(self):
		# Reset the environment
		state = self.env.reset()
		nA = self.action_size

		done = False

		while not done:
			act = self.policy(state)
			newstate, rew, done, _ = self.env.step(act)

			if len(self.repbuf) == self.options.replay_memory_size:
				self.repbuf.pop(0)
				self.repstates.pop(0)
				self.repqs.pop(0)

			self.repbuf.append((state, act, rew, newstate))
			self.repstates.append(state)
			self.repqs.append(rew)

			for i in range(self.options.batch_size):
				sampidx = np.random.choice(len(self.repbuf))
				(sj, aj, rj, nsj) = self.repbuf[sampidx]
				qj = np.zeros(len(rj))
				qnsj = self.target_model.predict(np.array([nsj,]))[0]

				for j in range(len(rj)):
					nsj = copy.copy(sj)
					nsj[j] += aj[j]
					if sampidx == len(self.repbuf)-1 and done:
						y = rj[j]
					else:
						qpred = self.target_model.predict(np.array([nsj,]))[0]
						y = rj[j] + self.options.gamma*np.max(qpred)
					qj[j] = y
				qj = self.options.beta*qj + (1-self.options.beta)*qnsj
				self.model.fit(np.array([sj,]), np.array([qj,]), epochs=1, verbose=0)


			self.nsteps += 1
			state = newstate
			if self.nsteps == self.options.update_target_estimator_every:
				self.nsteps = 0
				self.update_target_model()

		return [self.env.starterr, self.env.err, self.env.state]

	def eval_epsiode(self):
		evalpolicy = self.make_greedy_policy()
		state = self.env.reset()
		nA = self.action_size

		done = False
		states = [self.env.state]
		acts = []

		while not done:
			act = evalpolicy(state)
			newstate, rew, done, _ = self.env.step(act)

			state = newstate
			states.append(newstate)
			acts.append(act)


		return [self.env.err, states, acts]


	def make_greedy_policy(self):
		nA = self.action_size

		def policy_fn(state):
			act = self.bounded_action(self.get_action(state))
			if np.sum(act) != 0:
				raise ValueError('Sum of actions does not equal zero.')
			return act

		return policy_fn

	def make_epsilon_greedy_policy(self):
		nA = self.action_size

		def policy_fn(state):
			if np.random.rand() > self.options.epsilon:
				return self.bounded_action(self.get_action(state))
			else:
				act = np.random.randint([3]*self.env.nelems) - 1
				sumact = np.sum(act)
				if sumact > 0:
					cand = np.where(act > -1)[0]
					act[np.random.choice(cand, sumact, replace=False)] -= 1
				elif sumact < 0:
					cand = np.where(act  < 1)[0]
					act[np.random.choice(cand, -sumact, replace=False)] += 1
				if np.sum(act) != 0:
					raise ValueError('Sum of actions does not equal zero.')
				return self.bounded_action(act)

		return policy_fn

	def get_action(self, state):
		q = self.target_model.predict(np.array([state,]))[0]

		sortidx = np.argsort(q)
		qsort = q[sortidx]

		act = np.zeros(self.env.nelems, dtype=int)

		for i in range(self.env.nelems//2):
			lidx = self.env.nelems//2 - 1 - i
			ridx = self.env.nelems//2 + i
			dq = np.abs(qsort[lidx] - qsort[ridx])
			if dq < qthresh:
				act[lidx] = act[ridx] = 0
			else:
				act[lidx] = -1
				act[ridx] = 1

		return act[sortidx]

	def bounded_action(self, act):
		for i in range(self.env.nelems):
			if act[i] == 1 and self.env.state[i] == self.env.maxp - 1:
				act[i] = 0
			if act[i] == -1 and self.env.state[i] == 1:
				act[i] = 0
		sumact = np.sum(act)
		if sumact > 0:
			restr_act = copy.copy(act)
			idxs = np.where(self.env.state == 1)[0]
			for idx in idxs:
				restr_act[idx] = -1
			cand = np.where(restr_act > -1)[0]
			act[np.random.choice(cand, sumact, replace=False)] -= 1
		elif sumact < 0:
			restr_act = copy.copy(act)
			idxs = np.where(self.env.state == self.env.maxp - 1)[0]
			for idx in idxs:
				restr_act[idx] = 1
			cand = np.where(restr_act < 1)[0]
			act[np.random.choice(cand, -sumact, replace=False)] += 1
		if (np.sum(act) != 0):
			raise ValueError('Action space not bounded properly.')

		return act




class Options():
	def __init__(self, neps, layers, alpha, epsilon, replay_memory_size, batch_size, gamma, update_target_estimator_every, qthresh, beta):
		self.neps = neps
		self.layers = layers
		self.alpha = alpha
		self.epsilon = epsilon
		self.replay_memory_size = replay_memory_size
		self.batch_size = batch_size
		self.gamma = gamma
		self.update_target_estimator_every = update_target_estimator_every
		self.qthresh = qthresh
		self.beta = beta

env = PoissonSolverEnv()


neps = 10000
layers = [10, 5, 3]
alpha = 0.1
epsilon =  0.5
epsilon_decay =  0.993
batch_size = 64
replay_memory_size = 128
gamma = 0.95
update_target_estimator_every = 128
qthresh = 0
beta = 0.5

options = Options(neps, layers, alpha, epsilon, replay_memory_size, batch_size, gamma, update_target_estimator_every, qthresh, beta)

dqn = DQN(env, options)

data = []
print('Iteration, relative error, final state, epsilon')
for i in range(neps):
	[starterr, enderr, endstate] = dqn.train_episode()
	options.epsilon *= epsilon_decay

	dqn.update_options(options)

	print(i, enderr/starterr, endstate, options.epsilon)
	data.append([enderr/starterr, starterr, endstate, options.epsilon])

	if i%100 == 0:
		[evalerr, evalstates, evalacts] = dqn.eval_epsiode()
		with open('dqn_{0}.pickle'.format(i), mode='wb') as f:
			pickle.dump([dqn.model, dqn.target_model, data], f)
		print('Evaluation error: ', evalerr/starterr, 'Evaluation state: ', evalstates[-1])


