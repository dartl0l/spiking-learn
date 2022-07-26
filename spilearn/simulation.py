import numpy as np

import gym
import nest
import json
import pickle

from gym import wrappers

from tqdm import trange

from .evaluation import *
from .network import *
from .converter import *
from .reward import *
from .teacher import *


class Simulation():
    def __init__(self):
        pass

    def run(self):
        pass

class CartPoleSimulation(Simulation):
    
    def __init__(self, env, test_network, scaler, conv, reward, evaluation, teacher, settings, learning_rate=0.03):

        self.env = env
        self.scaler = scaler
        self.settings = settings
        self.conv = conv
        
        self.evaluation = evaluation
        self.custom_reward = reward
        self.network = test_network
        self.teacher = teacher
        
        self.spikes_length = 0
        self.learning_rate_default = learning_rate
        self.learning_rate = learning_rate
        self.teacher_amp = 1.0
        
    def simulate(self):
        nest.Simulate(self.settings['network']['h_time'])

        spikes = nest.GetStatus(self.network.spike_detector_out,
                                keys="events")[0]['times']
        senders = nest.GetStatus(self.network.spike_detector_out,
                                 keys="events")[0]['senders']
        self.spikes_length += spikes.size
        mask = spikes > self.time
        raw_latency = {
                          'spikes': spikes[mask],
                          'senders': senders[mask]
                         }

        raw_latency['spikes'] -= self.time
        return raw_latency

    def update_learning_rate(self, learning_rate):
        connection = nest.GetConnections(
            self.network.input_layer, 
            target=self.network.layer_out
        )
        nest.SetStatus(
            connection, 'lambda', 
            learning_rate
        )
        
    def set_noise(self, noise_freq):
        nest.SetStatus(
            self.poisson_layer,
            {
             'rate': noise_freq,
             'origin': 0.0
            }
        )

    def run_state(self, state):
        state = self.scaler.transform([state])
        state = self.conv.convert(state, [1])

        spike_dict, full_time = self.network.create_spike_dict(
            dataset=state['input'],
            delta=0.0)
        for spikes in spike_dict:
            spikes['spike_times'] += self.time

        self.network.set_input_spikes(
            spike_dict=spike_dict,
            spike_generators=self.network.input_generators)

        self.update_learning_rate(0.)

        raw_latency = self.simulate()
        self.time += full_time
        
        out_latency = self.evaluation.convert_latency([raw_latency])
        y_pred = self.evaluation.predict_from_latency(out_latency)
        return int(y_pred[0])

    def learn_states(self, states, actions, inhibit=False):
        states = self.scaler.transform(states)
        states = self.conv.convert(states, actions)

        states['input'] += self.time
        
        spike_dict, \
        full_time = self.network.create_spike_dict(
            dataset=states['input'],
            delta=0.0)

        teacher_dicts = self.teacher.create_teacher(
            input_spikes=states['input'],
            classes=np.array(actions),
            teachers=self.network.teacher_layer)
        
        self.network.set_input_spikes(
            spike_dict=spike_dict,
            spike_generators=self.network.input_generators)
        
        for teacher in teacher_dicts:
            teacher_dicts[teacher]['amplitude_values'] *= self.teacher_amp
            if inhibit:            
                teacher_dicts[teacher]['amplitude_values'] *= -1.0
                teacher_dicts[teacher]['amplitude_times'] -= self.settings['learning']['reinforce_delta_punish']
        self.network.set_teachers_input(
            teacher_dicts=teacher_dicts)

        self.update_learning_rate(self.learning_rate)

        raw_spikes = self.simulate()
        self.time += full_time
        return
    
    def run(self, n_episodes=100, n_states_max=10000):
        running_reward = 10
        running_cust_reward = 0
        running_reward_before_action = 0
        running_reward_after_action = 0
        weights_history = []
        reward_history = []
        lambdas = []
        amps = []
        
#         games_played = collections.deque(maxlen=5)
        learning_rate = lambda reward : self.learning_rate_default * (1 - reward / 200)

        counter = 0
        success = 0
        gamma = 0.99
        
        last_reward = 0
        
        self.time = self.settings['network']['start_delta']
        self.env.reset()

        nest.Simulate(self.time)

        t = trange(n_episodes)
        
        for e in t:
            state, ep_reward = self.env.reset(), 0  
            cust_reward = 0
            
            t.set_postfix({'reward': running_reward, 
                           'last games played': last_reward, 
                           'counter': counter,
                           'succces rate': success})
            
            if running_reward > 199.9:
                print("good job")
                t.close()
                break

            for ti in range(1, n_states_max):  # Don't infinite loop while learning
                reward_before_action = self.custom_reward(*state)
                action = self.run_state(state)

                new_state = [state]
                new_action = [action]
                
                state, reward, done, _ = self.env.step(action)
                reward_after_action = self.custom_reward(*state)

                cust_reward += reward_after_action
                ep_reward += reward

                if done:
                    if ti == 200: 
                        counter += 1
                    weights_history.append(self.network.save_weights(self.network.layers))
                    self.network.reset_spike_detectors()
                    last_reward = ep_reward
                    break
                
        
                punish = reward_before_action < reward_after_action
                delta_reward = reward_after_action - reward_before_action

                self.learning_rate = learning_rate(last_reward)
                self.teacher_amp = (1 - last_reward / 200)

                lambdas.append(self.learning_rate)
                amps.append(self.teacher_amp * self.settings['learning']['teacher_amplitude'])
                self.learn_states(new_state, new_action, inhibit=punish)
            
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            running_cust_reward = 0.05 * cust_reward + (1 - 0.05) * running_cust_reward
            reward_history.append((e, ti, running_reward, running_cust_reward))
            
            success = counter / (e + 1)
            
        return self.network.save_weights(self.network.layers), \
                np.array(weights_history), \
                np.array(reward_history).T, \
                np.array(lambdas), \
                np.array(amps)


class CartPoleSimulationTanh(CartPoleSimulation):

    def update_learning_rate(self, learning_rate):
        connection = nest.GetConnections(
            self.network.input_layer, 
            target=self.network.layer_out
        )
        nest.SetStatus(
            connection, 'a_plus', 
            learning_rate * self.settings['model']['syn_dict_stdp']['a_plus']
        )
        nest.SetStatus(
            connection, 'a_minus', 
            learning_rate * self.settings['model']['syn_dict_stdp']['a_minus']
        )
        
    def learn_states(self, states, actions, inhibit=False):
        states = self.scaler.transform(states)
        states = self.conv.convert(states, actions)

        states['input'] += self.time
        
        spike_dict, \
        full_time = self.network.create_spike_dict(
            dataset=states['input'],
            delta=0.0)

        teacher_dicts = self.teacher.create_teacher(
            input_spikes=states['input'],
            classes=np.array(actions),
            teachers=self.network.teacher_layer)
        
        self.network.set_input_spikes(
            spike_dict=spike_dict,
            spike_generators=self.network.input_generators)
        
        for teacher in teacher_dicts:
            teacher_dicts[teacher]['amplitude_values'] *= self.teacher_amp
            if inhibit:            
                teacher_dicts[teacher]['amplitude_values'] *= -1.0
                teacher_dicts[teacher]['amplitude_times'] -= self.settings['learning']['reinforce_delta_punish']
        self.network.set_teachers_input(
            teacher_dicts=teacher_dicts)

        self.update_learning_rate(self.learning_rate)
        
        raw_spikes = self.simulate()
        self.time += full_time
        return
    
    def run(self, n_episodes=100, n_states_max=10000):
        running_reward = 10
        running_cust_reward = 0
        running_reward_before_action = 0
        running_reward_after_action = 0
        weights_history = []
        reward_history = []
        lambdas = []
        amps = []
        
#         games_played = collections.deque(maxlen=5)
        learning_rate = lambda reward : self.learning_rate_default * (1 - reward / 200)

        counter = 0
        success = 0
        gamma = 0.99
        
        last_reward = 0
        
        self.time = self.settings['network']['start_delta']
        self.env.reset()

        nest.Simulate(self.time)

        t = trange(n_episodes)
        
        for e in t:
            state, ep_reward = self.env.reset(), 0  
            cust_reward = 0
            
            t.set_postfix({'reward': running_reward, 
                           'last games played': last_reward, 
                           'counter': counter,
                           'succces rate': success})
            
            if running_reward > 199.9:
                print("good job")
                t.close()
                break

            for ti in range(1, n_states_max):  # Don't infinite loop while learning
                reward_before_action = self.custom_reward(*state)
                action = self.run_state(state)

                new_state = [state]
                new_action = [action]
                
                state, reward, done, _ = self.env.step(action)
                reward_after_action = self.custom_reward(*state)

                cust_reward += reward_after_action
                ep_reward += reward

                if done:
                    if ti == 200: 
                        counter += 1
                    weights_history.append(self.network.save_weights(self.network.layers))
                    self.network.reset_spike_detectors()
                    last_reward = ep_reward
                    break
                
        
                punish = reward_before_action < reward_after_action
                delta_reward = reward_after_action - reward_before_action

#                 self.learning_rate = learning_rate(last_reward)
                self.learning_rate = learning_rate(last_reward)
                self.teacher_amp = (1 - last_reward / 200)

                lambdas.append(self.learning_rate)
                amps.append(self.teacher_amp * self.settings['learning']['teacher_amplitude'])
                self.learn_states(new_state, new_action, inhibit=punish)
            
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            running_cust_reward = 0.05 * cust_reward + (1 - 0.05) * running_cust_reward
            reward_history.append((e, ti, running_reward, running_cust_reward))
            
            success = counter / (e + 1)
           
        return self.network.save_weights(self.network.layers), \
                np.array(weights_history), \
                np.array(reward_history).T, \
                np.array(lambdas), \
                np.array(amps)


class CartPoleSimulationMem(CartPoleSimulation):

    def update_learning_rate(self, learning_rate):
        connection = nest.GetConnections(
            self.network.input_layer, 
            target=self.network.layer_out
        )
        nest.SetStatus(
            connection, 'alpha_plus', 
            learning_rate * self.settings['model']['syn_dict_stdp']['alpha_plus']
        )
        nest.SetStatus(
            connection, 'alpha_minus', 
            learning_rate * self.settings['model']['syn_dict_stdp']['alpha_minus']
        )

    def run(self, n_episodes=100, n_states_max=10000):
        running_reward = 10
        running_cust_reward = 0
        running_reward_before_action = 0
        running_reward_after_action = 0
        weights_history = []
        reward_history = []
        lambdas = []
        amps = []
        
#         games_played = collections.deque(maxlen=5)
        learning_rate = lambda reward : self.learning_rate_default * (1 - reward / 200)

        counter = 0
        success = 0
        gamma = 0.99
        
        last_reward = 0
        
        self.time = self.settings['network']['start_delta']
        self.env.reset()

        nest.Simulate(self.time)

        t = trange(n_episodes)
        
        for e in t:
            state, ep_reward = self.env.reset(), 0  
            cust_reward = 0
            
            t.set_postfix({'reward': running_reward, 
                           'last games played': last_reward, 
                           'counter': counter,
                           'succces rate': success})
            
            if running_reward > 199.9:
                print("good job")
                t.close()
                break

            for ti in range(1, n_states_max):  # Don't infinite loop while learning
                reward_before_action = self.custom_reward(*state)
                action = self.run_state(state)

                new_state = [state]
                new_action = [action]
                
                state, reward, done, _ = self.env.step(action)
                reward_after_action = self.custom_reward(*state)

                cust_reward += reward_after_action
                ep_reward += reward

                if done:
                    if ti == 200: 
                        counter += 1
                    weights_history.append(self.network.save_weights(self.network.layers))
                    self.network.reset_spike_detectors()
                    last_reward = ep_reward
                    break
                
        
                punish = reward_before_action < reward_after_action
                delta_reward = reward_after_action - reward_before_action

                self.learning_rate = learning_rate(last_reward)
#                 self.learning_rate = learning_rate(reward_after_action)
                self.teacher_amp = (1 - last_reward / 200)

                lambdas.append(self.learning_rate)
                amps.append(self.teacher_amp * self.settings['learning']['teacher_amplitude'])
                self.learn_states(new_state, new_action, inhibit=punish)
            
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            running_cust_reward = 0.05 * cust_reward + (1 - 0.05) * running_cust_reward
            reward_history.append((e, ti, running_reward, running_cust_reward))
            
            success = counter / (e + 1)
           
        return self.network.save_weights(self.network.layers), \
                np.array(weights_history), \
                np.array(reward_history).T, \
                np.array(lambdas), \
                np.array(amps)