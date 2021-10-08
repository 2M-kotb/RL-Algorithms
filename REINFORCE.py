import numpy as np
from itertools import count
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os
import random

# create policy network
class policy(nn.Module):
	def __init__(self, state_size,hidden_size,actions):
		super(policy, self).__init__()

		self.fc1 = nn.Linear(state_size,hidden_size)
		self.dropout = nn.Dropout(p=0.6)
		self.fc2 = nn.Linear(hidden_size,actions)


	def forward(self, state):
		x = self.fc1(state)
		x = self.dropout(x)
		x = F.relu(x)
		x = self.fc2(x)
		actions_probs = F.softmax(x, dim=1)
		return actions_probs


class REINFORCE():

	def __init__(self,state_size,hidden_size,actions):
		self.dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'policy')
		self.policy = policy(state_size, hidden_size, actions)
		self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

	def select_action(self, state):
		state = torch.from_numpy(state).float().unsqueeze(0)
		action_probs = self.policy(state)
		m = Categorical(action_probs)
		action = m.sample()
		action_log_prob = m.log_prob(action)
		return action.item(), action_log_prob

	def train(self,env,num_episodes, max_moves, render = False, gamma = 0.99):


		episode_total_reward = [] # store total reward of each episode

		for i_episode in range(num_episodes):

			rewards = [] # store reward at each step in the episode
			log_probs = [] # store episode actions log_prob
			ep_reward = 0 # compute the episode total reward
			
			state = env.reset()

			for t in range(max_moves):
				# select action
				action, a_log_prob = self.select_action(state)
				# take action and get the next state and reward
				state, reward, done, _ = env.step(action)

				if render:
					env.render()

				ep_reward += reward
				# store log_prob and reward
				rewards.append(reward)
				log_probs.append(a_log_prob)

				if done:
					env.close()
					break

			episode_total_reward.append(ep_reward)
			# compute the loss and update weights
			self.update(rewards,log_probs, gamma)

			print("episode:{}/{} =====> reward:{} ".format(i_episode+1,num_episodes,ep_reward))
		
		
		#save model policy network
		torch.save(self.policy.state_dict(),self.dir_path)

		#plot the training curve
		self.plot(episode_total_reward, num_episodes)

	def test(self,env,num_test):

		# load policy model
		self.policy.load_state_dict(torch.load(self.dir_path))

		self.policy.eval()

		for i in range(num_test):

			# use random seed for testing
			env.seed(random.randint(0,100))
			state = env.reset()
			rewards = 0
			while True:
				action, _ = self.select_action(state)
				state, reward, done, _ = env.step(action)
				rewards += reward
				env.render()
				if done: 
					env.close()
					break

			print("test:{} ===> reward:{}".format(i+1,rewards))



	def update(self,rewards,log_probs, gamma):
		returns = [] # compute return for each step
		policy_loss = [] # compute policy loss for each step
		R = 0
		# compute returns
		for r in rewards[::-1]:
			R = r + gamma * R
			returns.insert(0, R)

		returns = torch.tensor(returns) # convert to tensor
		eps = np.finfo(np.float32).eps.item()
		returns = (returns - returns.mean()) / (returns.std() + eps)

	    # compute policy loss at each step
		for log_prob, R in zip(log_probs, returns):
			policy_loss.append(-log_prob * R) # the minus sign cause we perform gradient ascent

		self.optimizer.zero_grad()
		policy_loss = torch.cat(policy_loss).sum()
		policy_loss.backward()
		self.optimizer.step()

		

	def plot(self, episodes_rewards, num_episodes):

		plt.style.use('seaborn')
		plt.plot(np.arange(num_episodes), episodes_rewards)
		plt.ylabel('reward', fontsize = 14)
		plt.xlabel('episodes', fontsize = 14)
		plt.title('reward vs episode', fontsize = 18, y = 1.03)
		plt.show()



if __name__ == '__main__':

	seed = 543
	env = gym.make('CartPole-v1')
	env.seed(seed)
	torch.manual_seed(seed)

	agent = REINFORCE(4,128,2)

	agent.train(env,1400,500)

	agent.test(env,2)









