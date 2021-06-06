class Simulation():
    
    def __init__(self, env, test_network, scaler, conv, reward, evaluation, teacher, settings):

        self.env = env
        self.scaler = scaler
        self.settings = settings
        self.conv = conv
        
        self.evaluation = evaluation
        self.custom_reward = reward
        self.network = test_network
        self.teacher = teacher
        
        self.spikes_length = 0
        self.lamb = settings['model']['syn_dict_stdp']['lambda']
#         self.teacher_amp = settings['learning']['teacher_amplitude']
        
    def simulate(self):
        nest.Simulate(settings['network']['h_time'])

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
    
    def run_state(self, state):
        state = self.scaler.transform([state])
        state = self.conv.convert(state, [1])

        spike_dict, full_time = self.network.create_spike_dict(
            dataset=state['input'],
            threads=self.settings['network']['num_threads'],
            delta=0.0)
        for spikes in spike_dict:
            spikes['spike_times'] += self.time

        self.network.set_input_spikes(
            spike_dict=spike_dict,
            spike_generators=self.network.input_generators)
        
        connection = nest.GetConnections(
            self.network.input_layer, 
            target=self.network.layer_out
        )
        nest.SetStatus(connection, 'lambda', 0.0)

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
            threads=self.settings['network']['num_threads'],
            delta=0.0)

#         self.teacher.settings['learning']['teacher_amplitude'] = self.teacher_amp
        teacher_dicts = self.teacher.create_teacher(
            input_spikes=states['input'],
            classes=np.array(actions),
            teachers=self.network.teacher_layer)
        
        self.network.set_input_spikes(
            spike_dict=spike_dict,
            spike_generators=self.network.input_generators)
        
        if inhibit:
            for teacher in teacher_dicts:
                teacher_dicts[teacher]['amplitude_values'] *= -1.0
                teacher_dicts[teacher]['amplitude_times'] -= settings['learning']['reinforce_delta_punish']
        self.network.set_teachers_input(
            teacher_dicts=teacher_dicts)

        connection = nest.GetConnections(
            self.network.input_layer, 
            target=self.network.layer_out
        )
        nest.SetStatus(
            connection, 'lambda', 
#             settings['model']['syn_dict_stdp']['lambda']
            self.lamb
        )
        
        raw_spikes = self.simulate()
        self.time += full_time

        # out_latency = self.evaluation.convert_latency([raw_latency])
        # y_pred = self.evaluation.predict_from_latency(out_latency)
        return # int(y_pred[0])
    
    def run(self, n_episodes=100, n_states_max=10000):
        running_reward = 10
        running_cust_reward = 0
        running_reward_before_action = 0
        running_reward_after_action = 0
        weights_history = []
        reward_history = []
        lambdas = []
        
#         games_played = collections.deque(maxlen=5)
        
        gamma = 0.99
        
        last_reward = 0
        
        self.time = self.settings['network']['start_delta']
        env.reset()

        nest.Simulate(self.time)

        t = trange(n_episodes)
        
        for e in t:
            state, ep_reward = env.reset(), 0  
            cust_reward = 0
            
            t.set_postfix({'reward': running_reward, 'last games played': last_reward, 'spikes length': self.spikes_length})
            
#             if np.mean(games_played) == 200: break
#             i = 0
#             trace_len = 3
#             states, actions, rewards = [None] * trace_len, [None] * trace_len, [None] * trace_len
            for ti in range(1, n_states_max):  # Don't infinite loop while learning
                reward_before_action = self.custom_reward(*state)
                action = self.run_state(state)
#                 action_ann = predict_ann_model([state])
#                 states[i] = state
#                 actions[i] = action
                new_state = [state]
                new_action = [action]
                
                state, reward, done, _ = env.step(action)
                reward_after_action = self.custom_reward(*state)

                cust_reward += reward_after_action
                running_reward_before_action = 0.05 * reward_before_action + (1 - 0.05) * running_reward_before_action
                running_reward_after_action = 0.05 * reward_after_action + (1 - 0.05) * running_reward_after_action
                ep_reward += reward
                
#                 not_done = 1.0 - int(done)
#                 advantage = reward + gamma * not_done * reward_after_action - reward_before_action

                if done:
                    weights_history.append(self.network.save_weights(self.network.layers))
                    last_reward = ep_reward
#                     games_played.append(last_reward)
                    break
                
#                 assert (advantage < 0) == (reward_before_action < reward_after_action)
#                 punish = advantage < 0
                punish = reward_before_action < reward_after_action
#                 punish = action != action_ann
#                 punish = False
#                 if punish:
#                     new_action = [int(not action)]
                
                lamb = lambda reward : self.settings['model']['syn_dict_stdp']['lambda'] * (1 - reward / 200)
#                 teach = lambda reward : self.settings['learning']['teacher_amplitude'] * (1 - reward / 200)
                
#                 self.teacher_amp = teach(reward)
                self.lamb = lamb(last_reward)
                lambdas.append(self.lamb)
#                 print(self.lamb)
#                 print(advantage, reward_before_action, reward_after_action, punish)
                self.learn_states(new_state, new_action, inhibit=punish)
            
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            running_cust_reward = 0.05 * cust_reward + (1 - 0.05) * running_cust_reward

#             print("{0}\t\t{1}\t\t{2}\t\t{3}".format(e, round(running_reward, 2), round(running_cust_reward, 2), ti))
            reward_history.append((e, ti, running_reward, running_cust_reward))
        return self.network.save_weights(self.network.layers), \
                np.array(weights_history), \
                np.array(reward_history).T, \
                np.array(lambdas)