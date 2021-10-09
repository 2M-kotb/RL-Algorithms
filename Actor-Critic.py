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


class Actor_Critic_Agent():

	def __init__(self,state_size,hidden_size,actions):
		self.dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'actor_critic')
		self.AC_model = Actor_Critic_Nets(state_size, hidden_size, actions)
		self.optimizer = optim.Adam(self.AC_model.parameters(), lr=5e-3)

	def select_action(self, state):
		state = torch.from_numpy(state).float().unsqueeze(0)
		# get actions_probs and state value from Actor-critic model
		actions_probs, value = self.AC_model(state)

		m = Categorical(actions_probs)
		action = m.sample()
		action_log_prob = m.log_prob(action)

		# return action, action_log_prob, state_value

		return action.item(), action_log_prob, value


	def train(self,env,num_episodes, max_moves, render = False, gamma = 0.99):


		episode_total_reward = [] # store total reward of each episode, used in plotting

		for i_episode in range(num_episodes):

			rewards = [] # store reward at each step in the episode
			log_probs = [] # store log_prob of the episode actions 
			values = [] # store values of the episode states
			ep_reward = 0 # compute the episode total reward
			
			state = env.reset()

			for t in range(max_moves):
				# select action
				action, a_log_prob, value = self.select_action(state)
				# take action and get the next state and reward
				state, reward, done, _ = env.step(action)

				if render:
					env.render()

				ep_reward += reward
				# store log_prob and reward
				rewards.append(reward)
				log_probs.append(a_log_prob)
				values.append(value)

				if done:
					env.close()
					break

			episode_total_reward.append(ep_reward)
			# compute the loss and update weights
			self.update(rewards,log_probs,values, gamma)

			print("episode:{}/{} =====> reward:{} ".format(i_episode+1,num_episodes,ep_reward))

		#save model policy network
		torch.save(self.AC_model.state_dict(),self.dir_path)

		#plot the training curve
		self.plot(episode_total_reward, num_episodes)


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

	def test(self,env,num_test):

		# load policy model
		self.AC_model.load_state_dict(torch.load(self.dir_path))

		self.AC_model.eval()

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


	def plot(self, episodes_rewards, num_episodes):

		plt.style.use('seaborn')
		plt.plot(np.arange(num_episodes), episodes_rewards)
		plt.ylabel('reward', fontsize = 14)
		plt.xlabel('episodes', fontsize = 14)
		plt.title('reward vs episode', fontsize = 18, y = 1.03)
		plt.show()



if __name__ == '__main__':

	seed = 543
	env = gym.make('CartPole-v1')   #CartPole-v1
	env.seed(seed)
	torch.manual_seed(seed)

	agent = Actor_Critic_Agent(4,128,2)

	agent.train(env,1400,500)

	agent.test(env,2)







		