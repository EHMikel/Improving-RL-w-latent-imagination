import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.distributions import Categorical  
from torch.autograd import Variable    

class PolicyNet(nn.Module): 
    def __init__(self, input_size, num_actions):
        super(PolicyNet, self).__init__() 

        # arquitectura de la red 
        self.fc1 = nn.Linear(in_features=input_size, out_features=16*input_size)
        self.fc2 = nn.Linear(in_features=16*input_size, out_features=32*input_size)
        self.fc3 = nn.Linear(in_features= 32*input_size, out_features=num_actions)

    # Se definen las operaciones del forward
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)  
        
        return x 
    
def process_state(state): 
    state = torch.from_numpy(state).float()
    return Variable(state)

if __name__ == '__main__': 

    episodes = 10_000
    max_steps_in_episode = 1000
    episode_duration = []             # almacenar la duracion de los episodios
    episode_total_rewards = []        # almacenar la recompensa total de cada episodio
    model_path = 'LunarLander_pg.pth' # guardar el modelo en este directorio

    # hiper parámetros del entrenamiento
    learning_rate = 0.001
    gamma = 0.99
    batch_size = 32

    # se define el enviroment y la policy
    env = gym.make(
        "LunarLander-v2",
        continuous = False,
        gravity = -10.0,
        enable_wind= False,
        wind_power = 15.0,
        turbulence_power = 1.5,
        )

    policy_net = PolicyNet(input_size=env.observation_space.shape[0], num_actions=env.action_space.n)

    # optimizador 
    optimizer = torch.optim.Adam(
        params=policy_net.parameters(), 
        lr= learning_rate
    )

    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0

    # print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)

    # ENTRENAMIENTO
    # bucle de episodios
    for eps in range(episodes): # 

        print(f'\nInicializando el episodio: {eps}')
        state = env.reset()
        state = process_state(state=state[0]).to(device)

        episode_rewards = []
        # bucle de trayectorias
        for t in range(max_steps_in_episode): 

            # seleccionar una accion dado el estado y la policy actual
            probs = policy_net(state)
            m = Categorical(probs= probs)
            action = m.sample()
            action = action.item()

            # ejecuttamos un step en el entorno con nuestra action
            next_state, reward, done, truncated,  _ = env.step(action=action)
            episode_rewards.append(reward)

            # almacenamos experiencia
            state_pool.append(state)
            action_pool.append(float(action))
            
            # procesamos el siguiente estado
            state = process_state(next_state).to(device)
            steps += 1

            # # si llegamos al estado terminal se rompe el bucle
            if done:   
                print('done!')
                break
        
        # si el episoido termina por llegar al limite de steps
        else: print('not done!')

        episode_duration.append(t+1)
        episode_total_rewards.append(np.sum(episode_rewards))
        print(f'Finaliza el episodio: {eps}\n\tduración: {t} steps\n\trecompensa total: {np.sum(episode_rewards)}\n\n')

        # calcular la recompensa acumulada para cada step EN CADA EPISODIO
        discount_reward = 0
        for n in reversed(range(len(episode_rewards))): 
            discount_reward = episode_rewards[n] + gamma*discount_reward
            episode_rewards[n] = discount_reward

        # normalizamos la recompensa DE CADA EPISODIO
        reward_mean = np.mean(episode_rewards)
        reward_std = np.std(episode_rewards)
        norm_episode_rewards = [(episode_rewards[n] - reward_mean)/(reward_std+ 1e-9) for n in range(len(episode_rewards))]
        
        # añadimos los discount rewards normalizados por episodio y descontados por episodio
        reward_pool.extend(norm_episode_rewards)

        # update policy; actualizamos la policy cada 5 trayectorias recopiladas
        if eps >0 and eps% batch_size == 0: 
            print('\n\tActualizando la policy...')

            # actualizar la policy
            optimizer.zero_grad()
            total_loss = 0
            for n in range(steps): 
                
                state = state_pool[n]
                action = Variable(torch.tensor(action_pool[n], dtype=torch.float32)).to(device)

                reward = reward_pool[n]
                probs = policy_net(state)           # le damos a la policy el nuevo estado 
                m = Categorical(probs=probs)        # sacamos las probs para cada accion
                loss = -m.log_prob(action)*reward   # negative score  function x reward; calculamos la pérdida de cada accion por su recompensa
                total_loss += loss
                
            # calcular gradientes
            total_loss.backward()

            # actualizamos los pesos en direccion de los gradientes
            optimizer.step()  
            print('\tPolicy actualizada con éxito!\n')   

            # limpiamos la memoria (cada vez que termino un entreno)
            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0
    
    # Guardar el modelo
    torch.save(policy_net.state_dict(), model_path)

    # curvas de entrenamiento: duración de episodios y recompensas
    # plt.figure(figsize=(10, 8))
    # plt.plot(range(len(episode_duration)), episode_duration)
    # plt.title('Duración de los epsiodios en LunarLander Policy gradients')
    # plt.grid()
    # plt.show()

    # plt.figure(figsize= (10, 8))
    # plt.plot(range(len(episode_total_rewards)), episode_total_rewards, label= 'policy grandients')
    # plt.axhline(y= 200, color= 'orange', linestyle= '-', label= 'won episodes')
    # plt.title('Recomensas promedio en cada episdio; LunarLander Policy gradients')
    # plt.legend(loc= 'best')
    # plt.grid()
    # plt.show()

    # EVALUAR EL MODELO
    evaluation_eps = 10
    max_steps_in_evaluation = 400

    env = gym.make(
        "LunarLander-v2",
        continuous = False,
        gravity = -10.0,
        enable_wind= False,
        wind_power = 15.0,
        turbulence_power = 1.5,
        render_mode ='human'
    )

    model = PolicyNet(input_size=env.observation_space.shape[0], num_actions=env.action_space.n)
    model.load_state_dict(torch.load(model_path))

    ##bucle de episodios
    for eps in range(evaluation_eps): 
        print('Inicializa episodio: ', eps)
        state = env.reset()
        state = process_state(state=state[0]).to(device)
        episode_rewards = []

    # bucle de trayectorias
        for t in range(max_steps_in_evaluation): 

            #seleccionar una accion dado el estado y la policy actual
            probs = model(state)
            m = Categorical(probs= probs)
            action = m.sample()
            action = action.item()

            env.render()

            # ejecuttamos un step en el entorno con nuestra action
            next_state, reward, done, truncated,  _ = env.step(action=action)
            episode_rewards.append(reward)

            # actualizamos el estado
            state = process_state(next_state).to(device)

            if done: break
        print(f'Recompensa total del episodio: {np.sum(episode_rewards)} en {t+1} steps\n')