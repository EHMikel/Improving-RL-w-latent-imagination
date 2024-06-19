import numpy as np 
import pandas as pd 

def discount_rewards(episode_rewards:list, gamma= 0.99)->list: 
    # calcular la recompensa acumulada para cada step EN CADA EPISODIO
    discounted_eps_rewards= [0]*len(episode_rewards) 
    discount_reward = 0
    for n in reversed(range(len(episode_rewards))): 
        discount_reward = episode_rewards[n] + gamma*discount_reward
        discounted_eps_rewards[n] = discount_reward
        return discounted_eps_rewards

def normalize_rewards(episode_rewards:list)->list: 
    # normalizamos la recompensa
    reward_mean = np.mean(episode_rewards)
    reward_std = np.std(episode_rewards)
    norm_episode_rewards = [(episode_rewards[n] - reward_mean)/(reward_std+ 1e-9) \
                            for n in range(len(episode_rewards))]
    return norm_episode_rewards
    
