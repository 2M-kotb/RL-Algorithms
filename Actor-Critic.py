import numpy as np
from itertools import count
import gym
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os
import random
import pickle
import seaborn as sns


# create Actor-Critic model

class Actor_Critic_Nets(nn.Module):

	def __init__(self, state_size, hidden_size, action_size):
		super(Actor_Critic_Nets,self).__init__()

		
		self.fc1 = nn.Linear(state_size, hidden_size)
		self.dropout = nn.Dropout(p=0.6)

		# Actor Network
		self.actor_head = nn.Linear(hidden_size, action_size)

		# Critic Network
		self.critic_head = nn.Linear(hidden_size, 1)

	def forward(self,state):
		
		x = self.fc1(state)
		x = F.relu(self.dropout(x))

		# actor head
		actions_probs = F.softmax(self.actor_head(x), dim=-1)

		# critic head
		value = self.critic_head(x)

		return actions_probs, value

# create Actor-critic Agent
class Actor_Critic_Agent():

	def __init__(self,state_size,hidden_size,actions):
		self.dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'results')
		self.AC_model = Actor_Critic_Nets(state_size, hidden_size, actions)
		self.optimizer = optim.Adam(self.AC_model.parameters(), lr=5e-3)
		self.AC_model.train() # training mode

	def select_action(self, state):
		state = torch.from_numpy(state).float().unsqueeze(0)
		# get actions_probs and state value from Actor-critic model
		actions_probs, value = self.AC_model(state)

		m = Categorical(actions_probs)
		action = m.sample()
		action_log_prob = m.log_prob(action)

		# return action, action_log_prob, state_value

		return action.item(), action_log_prob, value


	def train(self,env,num_episodes, max_moves, seed, train_diff_seeds = False, render = False, gamma = 0.99):


		episodes_total_return = [] # store total return of each episode, used in plotting

		for i_episode in range(num_episodes):

			rewards = [] # store reward at each step in the episode
			log_probs = [] # store log_prob of the episode actions 
			values = [] # store values of the episode states
			ep_return = 0 # compute the episode total return
			
			state = env.reset()

			for t in range(max_moves):
				# select action
				action, a_log_prob, value = self.select_action(state)
				# take action and get the next state and reward
				state, reward, done, _ = env.step(action)

				if render:
					env.render()

				ep_return += reward
				# store log_prob and reward
				rewards.append(reward)
				log_probs.append(a_log_prob)
				values.append(value.squeeze(0))

				if done:
					env.close()
					break

			episodes_total_return.append(ep_return)
			# compute the loss and update weights
			self.update(rewards,log_probs,values, gamma)

			if not train_diff_seeds:

				if(i_episode+1) % 10 == 0:

					print("episode:{}/{} =====> reward:{} ".format(i_episode+1,num_episodes,ep_return))

		#save model policy network
		model_dir = os.path.join(self.dir_path,"AC_model_{}".format(seed))
		torch.save(self.AC_model.state_dict(),model_dir)

		if not train_diff_seeds:
			#plot the training curve
			self.plot(episodes_total_return, num_episodes)
		else:
			# save the episodes returns for every seeds
			file = "returns_{}.pkl".format(seed)
			open_file = open(file, "wb")
			pickle.dump(episodes_total_return, open_file)
			open_file.close()


	def update(self,rewards,log_probs,values, gamma):
		returns = [] # compute return for each step
		actor_loss = [] # compute actor loss for each step
		critic_loss = [] # compute critic loss for each step
		R = 0

		# compute returns
		for r in rewards[::-1]:
			R = r + gamma * R
			returns.insert(0, R)

		returns = torch.tensor(returns) # convert to tensor
		eps = np.finfo(np.float32).eps.item()
		returns = (returns - returns.mean()) / (returns.std() + eps)

		for value, R, log_prob in zip(values, returns, log_probs):
			#compute advantage = G(t) - V(t)
			advantage = R - value.item()
			# calculate actor (policy) loss 
			actor_loss.append(-log_prob * advantage)
			# calculate critic loss as MSE, but we use a soft version of MSE called soft_l1_loss
			critic_loss.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
		self.optimizer.zero_grad()

		# sum up all the values of critic_loss and actor_loss
		loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()

		# perform backprop
		loss.backward()
		self.optimizer.step()

	def test(self,env,num_test,seed):

		# load policy model
		model_dir = os.path.join(self.dir_path,"AC_model_{}".format(seed))
		self.AC_model.load_state_dict(torch.load(model_dir))

		self.AC_model.eval() # testing mode

		for i in range(num_test):

			# use random seed for testing
			env.seed(random.randint(0,100))
			state = env.reset()
			rewards = 0
			while True:
				action, _, _ = self.select_action(state)
				state, reward, done, _ = env.step(action)
				rewards += reward
				env.render()
				if done: 
					env.close()
					break

			print("test:{} ===> reward:{}".format(i+1,rewards))


	def plot(self, episodes_returns, num_episodes):

		# compute average reward of window size = 10
		smoothed_returns = pd.Series.rolling(pd.Series(episodes_returns), 10).mean()
		smoothed_returns = [elem for elem in smoothed_returns]

		plt.style.use('seaborn')
		plt.plot(np.arange(num_episodes), smoothed_returns, color = "red")
		plt.ylabel('Average episodes return', fontsize = 14)
		plt.xlabel('episodes', fontsize = 14)
		plt.title('return vs episode', fontsize = 18, y = 1.03)
		plt.show()

	def plot_diff_seeds(self,seeds):

		# read saved episodes_returns for diff seeds
		data = [] 
		for seed in seeds:
			file_name = "returns_{}.pkl".format(seed.item())
			open_file = open(file_name, "rb")
			l = pickle.load(open_file)
			# convert to dataframe
			df = pd.DataFrame({"episodes":np.arange(len(l)),"AvgEpRet":l}) 
			data.append(df)
			open_file.close()

		# smooth returns by averaging over 10 episodes
		for i in range(len(data)):
			data[i]["AvgEpRet"] = data[i]["AvgEpRet"].rolling(10).mean()

		data = pd.concat(data, ignore_index=True) # concatenate into one dataframe

		# plot the mean episodes returns for the diff seeds and the standard deviation
		plt.style.use('seaborn')
		sns.lineplot(x="episodes", y="AvgEpRet" , data=data, color="orange", ci="sd")
		plt.show()




if __name__ == '__main__':

	"""
	First: fix seed and search for the best Hyperparameters
	"""

	# seed = 543
	# env = gym.make('CartPole-v1')   #CartPole-v1
	# env.seed(seed)
	# torch.manual_seed(seed)

	# agent = Actor_Critic_Agent(4,128,2)
	

	# agent.train(env,200,500,seed)

	# agent.test(env,2,seed)

	#-----------------------------------------------------
	"""
	Second: Train with different seeds to evaluate Actor-Critic agent
	"""
	Train with 3 different seeds
	seeds = np.random.randint(0,100,size=3)
	print("start training with {} seeds".format(len(seeds)))

	for seed in seeds:
		env.seed(seed.item())
		torch.manual_seed(seed)
		agent = Actor_Critic_Agent(4,128,2)
		agent.train(env,1400,500,seed,True)
		print("Finish training with seed:",seed)

	# plot the mean episodes rewards over diff seeds
	agent.plot_diff_seeds(seeds)









		