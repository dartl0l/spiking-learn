# coding: utf-8

import numpy as np


class Teacher:
    def __init__(self, settings):
        self.settings = settings
        
    def create_teacher(self, input_spikes, classes, teachers):  # Network
        h = self.settings['network']['h']
        h_time = self.settings['network']['h_time']
        start = self.settings['network']['start_delta']
        reinforce_time = self.settings['learning']['reinforce_time']
        reinforce_delta = self.settings['learning']['reinforce_delta']
        teacher_amplitude = self.settings['learning']['teacher_amplitude']

        full_time = len(input_spikes) * h_time + start
        times = np.arange(start, full_time, h_time)
        pattern_start_times = np.expand_dims(np.tile(times, (len(input_spikes[0]), 1)).T, axis=2)
        assert len(input_spikes) == len(pattern_start_times)

        spike_times = np.add(input_spikes, pattern_start_times)
        stimulation_start = np.nanmin(spike_times, axis=1) + reinforce_delta
        stimulation_end = stimulation_start + reinforce_time + 2 * h
        assert len(stimulation_start) == len(spike_times)

        teacher_dict = self.create_teacher_dict(stimulation_start, stimulation_end,
                                                classes, teachers, teacher_amplitude)
        return teacher_dict

    def create_teacher_dict(self, stimulation_start, stimulation_end, classes, teachers, teacher_amplitude):
        single_neuron = self.settings['topology']['n_layer_out'] == 1
        teacher_dict = {}
        for teacher in teachers:
            teacher_dict[teacher] = {
                'amplitude_times': np.ndarray([]),
                'amplitude_values': np.ndarray([])
            }
        for cl in set(classes):
            class_mask = classes == cl
            stimulation_start_current = stimulation_start[class_mask]
            stimulation_end_current = stimulation_end[class_mask]
            current_teacher_id = teachers[0] if single_neuron else teachers[cl]
            amplitude_times = np.stack((stimulation_start_current,
                                        stimulation_end_current), axis=-1).flatten()
            amplitude_values = np.stack((np.full_like(stimulation_start_current, teacher_amplitude),
                                         np.zeros_like(stimulation_end_current)), axis=-1).flatten()
            assert len(amplitude_times) == len(stimulation_start_current) * 2
            assert len(amplitude_values) == len(stimulation_end_current) * 2
            teacher_dict[current_teacher_id]['amplitude_times'] = amplitude_times
            teacher_dict[current_teacher_id]['amplitude_values'] = amplitude_values
        return teacher_dict


class TeacherMax(Teacher):
    def __init__(self, settings):
        self.settings = settings
        
    def create_teacher(self, input_spikes, classes, teachers):  # Network
        h = self.settings['network']['h']
        h_time = self.settings['network']['h_time']
        start = self.settings['network']['start_delta']
        reinforce_time = self.settings['learning']['reinforce_time']
        reinforce_delta = self.settings['learning']['reinforce_delta']
        teacher_amplitude = self.settings['learning']['teacher_amplitude']

        full_time = len(input_spikes) * h_time + start
        times = np.arange(start, full_time, h_time)
        pattern_start_times = np.expand_dims(np.tile(times, (len(input_spikes[0]), 1)).T, axis=2)
        assert len(input_spikes) == len(pattern_start_times)

        spike_times = np.add(input_spikes, pattern_start_times)
        stimulation_start = np.nanmax(spike_times, axis=1) + reinforce_delta
        stimulation_end = stimulation_start + reinforce_time + 2 * h
        assert len(stimulation_start) == len(spike_times)

        teacher_dict = self.create_teacher_dict(stimulation_start, stimulation_end,
                                                classes, teachers, teacher_amplitude)
        return teacher_dict


class TeacherFrequency(Teacher):
    
    def __init__(self, settings):
        super(TeacherFrequency, self).__init__(settings)
        
    def create_teacher(self, input_spikes, classes, teachers):  # Network
        h = self.settings['network']['h']
        h_time = self.settings['network']['h_time']
        epochs = self.settings['learning']['epochs']
        d_time = self.settings['network']['start_delta']
        single_neuron = self.settings['topology']['n_layer_out'] == 1
        reinforce_time = self.settings['learning']['reinforce_time']
        reinforce_delta = self.settings['learning']['reinforce_delta']
        teacher_amplitude = self.settings['learning']['teacher_amplitude']

        teacher_dicts = {}
        for teacher in teachers:
            teacher_dicts[teacher] = {
                                      'amplitude_times': [],
                                      'amplitude_values': []
                                     }
        # TODO
        # calc amplitude times one time and concatenate
        for _ in range(epochs):
            for spikes, cl in zip(input_spikes, classes):
                current_teacher_id = teachers[0] if single_neuron else teachers[cl]
                current_teacher = teacher_dicts[current_teacher_id]
                start_of_stimulation = d_time \
                    + reinforce_delta
                end_of_stimulation = start_of_stimulation \
                    + reinforce_time \
                    + h
                current_teacher['amplitude_times'].append(start_of_stimulation)
                current_teacher['amplitude_times'].append(end_of_stimulation)

                current_teacher['amplitude_values'].append(teacher_amplitude)
                current_teacher['amplitude_values'].append(0.0)
                d_time += h_time
        return teacher_dicts


class TeacherFull(Teacher):

    def __init__(self, settings):
        super(TeacherFull, self).__init__(settings)

    def create_teacher_dict(self, stimulation_start, stimulation_end, classes, teachers, teacher_amplitude):
        single_neuron = self.settings['topology']['n_layer_out'] == 1
        epochs = self.settings['learning']['epochs']
        classes_full = np.tile(classes, epochs)
        teacher_dict = {}
        for teacher in teachers:
            teacher_dict[teacher] = {
                'amplitude_times': np.ndarray([]),
                'amplitude_values': np.ndarray([])
            }
        for cl in set(classes):
            class_mask = classes_full == cl
            stimulation_start_current = stimulation_start[class_mask]
            stimulation_end_current = stimulation_end[class_mask]
            current_teacher_id = teachers[0] if single_neuron else teachers[cl]
            amplitude_times = np.stack((stimulation_start_current,
                                        stimulation_end_current), axis=-1).flatten()
            amplitude_values = np.stack((np.full_like(stimulation_start_current, teacher_amplitude),
                                         np.zeros_like(stimulation_end_current)), axis=-1).flatten()
            assert len(amplitude_times) == len(stimulation_start_current) * 2
            assert len(amplitude_values) == len(stimulation_end_current) * 2
            teacher_dict[current_teacher_id]['amplitude_times'] = amplitude_times
            teacher_dict[current_teacher_id]['amplitude_values'] = amplitude_values
        return teacher_dict


class TeacherInhibitory(Teacher):
    
    def __init__(self, settings):
        super(TeacherInhibitory, self).__init__(settings)
        
    def create_teacher_dict(self, stimulation_start, stimulation_end, classes, teachers, teacher_amplitude):
        single_neuron = self.settings['topology']['n_layer_out'] == 1
        epochs = self.settings['learning']['epochs']
        classes_full = np.tile(classes, epochs)
        teacher_dict = {}
        for teacher in teachers:
            teacher_dict[teacher] = {
                'amplitude_times': np.ndarray([]),
                'amplitude_values': np.ndarray([])
            }
        for cl in set(classes):
            current_teacher_id = teachers[0] if single_neuron else teachers[cl]
            class_mask = classes_full == cl
            stimulation_start_current = stimulation_start[class_mask]
            stimulation_end_current = stimulation_end[class_mask]
            amplitude_times = np.stack((stimulation_start_current,
                                        stimulation_end_current), axis=-1).flatten()
            amplitude_values_pos = np.stack((np.full_like(stimulation_start_current, teacher_amplitude),
                                             np.zeros_like(stimulation_end_current)), axis=-1).flatten()
            amplitude_values_neg = np.stack((np.full_like(stimulation_start_current, -teacher_amplitude),
                                             np.zeros_like(stimulation_end_current)), axis=-1).flatten()
            assert len(amplitude_times) == len(stimulation_start[class_mask]) * 2
            assert len(amplitude_values_pos) == len(stimulation_start[class_mask]) * 2
            assert len(amplitude_values_neg) == len(stimulation_start[class_mask]) * 2
            for teacher_id in teachers:
                if current_teacher_id != teacher_id:
                    teacher_dict[teacher_id]['amplitude_times'] = amplitude_times
                    teacher_dict[teacher_id]['amplitude_values'] = amplitude_values_neg
                # else:
                #     teacher_dict[teacher_id]['amplitude_values'] = amplitude_values_pos
        return teacher_dict


class ReinforceTeacher(Teacher):
    def __init__(self, settings):
        self.settings = settings
        
    def create_teacher(self, input_spikes, classes, teachers):  # Network
#         print("prepare teacher")
        h = self.settings['network']['h']
        h_time = self.settings['network']['h_time']
#         start = self.settings['network']['start_delta']
        reinforce_time = self.settings['learning']['reinforce_time']
        reinforce_delta = self.settings['learning']['reinforce_delta']
        teacher_amplitude = self.settings['learning']['teacher_amplitude']

        full_time = len(input_spikes) * h_time  # + start
        times = np.arange(0, full_time, h_time)
        pattern_start_times = np.expand_dims(np.tile(times, (len(input_spikes[0]), 1)).T, axis=2)
        assert len(input_spikes) == len(pattern_start_times)

        spike_times = np.add(input_spikes, pattern_start_times)
        stimulation_start = np.nanmin(spike_times, axis=1) + reinforce_delta
        stimulation_end = stimulation_start + reinforce_time + 2 * h
        assert len(stimulation_start) == len(spike_times)

        teacher_dict = self.create_teacher_dict(stimulation_start, stimulation_end,
                                                classes, teachers, teacher_amplitude)
        return teacher_dict

    def create_teacher_dict(self, stimulation_start, stimulation_end, classes, teachers, teacher_amplitude):
        single_neuron = self.settings['topology']['n_layer_out'] == 1
        teacher_dict = {}
        for teacher in teachers:
            teacher_dict[teacher] = {
                'amplitude_times': np.ndarray([]),
                'amplitude_values': np.ndarray([])
            }
              
        for cl in range(len(teachers)):
            class_mask = classes == cl

            stimulation_start_current = stimulation_start[class_mask]
            stimulation_end_current = stimulation_end[class_mask]
            current_teacher_id = teachers[0] if single_neuron else teachers[cl]
            
            amplitude_times = np.stack((stimulation_start_current,
                                        stimulation_end_current), axis=-1).flatten()
            amplitude_values = np.stack((np.full_like(stimulation_start_current, teacher_amplitude),
                                         np.zeros_like(stimulation_end_current)), axis=-1).flatten()
            assert len(amplitude_times) == len(stimulation_start_current) * 2
            assert len(amplitude_values) == len(stimulation_end_current) * 2
            teacher_dict[current_teacher_id]['amplitude_times'] = amplitude_times
            teacher_dict[current_teacher_id]['amplitude_values'] = amplitude_values
        return teacher_dict

