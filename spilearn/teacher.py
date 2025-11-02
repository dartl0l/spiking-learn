# coding: utf-8

import nest
import numpy as np


class Teacher:
    """
    Teacher signal generator for EpochNetwork class
    """

    def __init__(
        self,
        n_layer_out,
        teacher_amplitude,
        reinforce_delta=0.0,
        reinforce_time=0.0,
        start=50,
        h_time=50,
        h=0.01,
        use_min=True,
    ):
        self.h = h
        self.h_time = h_time
        self.start = start
        self.use_min = use_min

        self.reinforce_time = reinforce_time
        self.reinforce_delta = reinforce_delta
        self.teacher_amplitude = teacher_amplitude

        self.n_layer_out = n_layer_out

        self.teacher_layer = None

    def reset_teachers(self):
        self.teacher_layer.set({'amplitude_times': [], 'amplitude_values': []})

    def set_teachers_input(self, teacher_dicts):
        self.teacher_layer.set(list(teacher_dicts.values()))

    def create_teacher_layer(self):
        self.teacher_layer = nest.Create('step_current_generator', self.n_layer_out)

    def connect_teacher(self, layer_out):
        nest.Connect(
            self.teacher_layer, layer_out, 'one_to_one', syn_spec='static_synapse'
        )

    def create_teacher(self, input_spikes, classes):
        full_time = len(input_spikes) * self.h_time + self.start
        times = np.arange(self.start, full_time, self.h_time)
        pattern_start_times = np.expand_dims(
            np.tile(times, (len(input_spikes[0]), 1)).T, axis=2
        )
        assert len(input_spikes) == len(pattern_start_times)

        spike_times = np.add(input_spikes, pattern_start_times)
        stimulation_start = (
            np.nanmin(spike_times, axis=1) if self.use_min else times
        ) + self.reinforce_delta
        stimulation_end = stimulation_start + self.reinforce_time + 2 * self.h
        assert len(stimulation_start) == len(spike_times)

        teacher_dict = self.create_teacher_dict(
            stimulation_start,
            stimulation_end,
            classes,
            self.teacher_layer,
            self.teacher_amplitude,
        )
        return teacher_dict

    def create_teacher_dict(
        self, stimulation_start, stimulation_end, classes, teachers, teacher_amplitude
    ):
        single_neuron = self.n_layer_out == 1
        teacher_dict = {}
        for teacher in teachers.get('global_id'):
            teacher_dict[teacher] = {
                'amplitude_times': np.ndarray([]),
                'amplitude_values': np.ndarray([]),
            }
        for cl in np.unique(classes):
            class_mask = classes == cl
            stimulation_start_current = stimulation_start[class_mask]
            stimulation_end_current = stimulation_end[class_mask]
            current_teacher_id = teachers[0] if single_neuron else teachers[cl]
            amplitude_times = np.stack(
                (stimulation_start_current, stimulation_end_current), axis=-1
            ).flatten()
            amplitude_values = np.stack(
                (
                    np.full_like(stimulation_start_current, teacher_amplitude),
                    np.zeros_like(stimulation_end_current),
                ),
                axis=-1,
            ).flatten()
            assert len(amplitude_times) == len(stimulation_start_current) * 2
            assert len(amplitude_values) == len(stimulation_end_current) * 2
            teacher_dict[current_teacher_id.get('global_id')]['amplitude_times'] = (
                amplitude_times
            )
            teacher_dict[current_teacher_id.get('global_id')]['amplitude_values'] = (
                amplitude_values
            )
        return teacher_dict


class TeacherPool(Teacher):
    def __init__(self, pool_size, **kwargs):
        super(TeacherPool, self).__init__(**kwargs)
        self.pool_size = pool_size

    def create_teacher_dict(
        self, stimulation_start, stimulation_end, classes, teachers, teacher_amplitude
    ):
        teacher_dict = {}
        for teacher in teachers.get('global_id'):
            teacher_dict[teacher] = {
                'amplitude_times': np.ndarray([]),
                'amplitude_values': np.ndarray([]),
            }

        assert self.pool_size * len(set(classes)) == self.n_layer_out, (
            f'{self.pool_size}, {len(set(classes))}, {self.pool_size * len(set(classes))}, {self.n_layer_out}'
        )

        for cl in np.unique(classes):
            class_mask = classes == cl
            stimulation_start_current = stimulation_start[class_mask]
            stimulation_end_current = stimulation_end[class_mask]
            current_teacher_ids = teachers[
                cl * self.pool_size : cl * self.pool_size + self.pool_size
            ]
            amplitude_times = np.stack(
                (stimulation_start_current, stimulation_end_current), axis=-1
            ).flatten()
            amplitude_values = np.stack(
                (
                    np.full_like(stimulation_start_current, teacher_amplitude),
                    np.zeros_like(stimulation_end_current),
                ),
                axis=-1,
            ).flatten()
            assert len(amplitude_times) == len(stimulation_start_current) * 2
            assert len(amplitude_values) == len(stimulation_end_current) * 2
            for current_teacher_id in current_teacher_ids:
                teacher_dict[current_teacher_id.get('global_id')]['amplitude_times'] = (
                    amplitude_times
                )
                teacher_dict[current_teacher_id.get('global_id')][
                    'amplitude_values'
                ] = amplitude_values
        return teacher_dict


class TeacherMax(Teacher):
    def __init__(self, **kwargs):
        super(TeacherMax, self).__init__(**kwargs)

    def create_teacher(self, input_spikes, classes):
        full_time = len(input_spikes) * self.h_time + self.start
        times = np.arange(self.start, full_time, self.h_time)
        pattern_start_times = np.expand_dims(
            np.tile(times, (len(input_spikes[0]), 1)).T, axis=2
        )
        assert len(input_spikes) == len(pattern_start_times)

        spike_times = np.add(input_spikes, pattern_start_times)
        stimulation_start = np.nanmax(spike_times, axis=1) + self.reinforce_delta
        stimulation_end = stimulation_start + self.reinforce_time + 2 * self.h
        assert len(stimulation_start) == len(spike_times)

        teacher_dict = self.create_teacher_dict(
            stimulation_start,
            stimulation_end,
            classes,
            self.teacher_layer,
            self.teacher_amplitude,
        )
        return teacher_dict


class TeacherFrequency(Teacher):
    def __init__(self, epochs, **kwargs):
        super(TeacherFrequency, self).__init__(**kwargs)
        self.epochs = epochs

    def create_teacher(self, input_spikes, classes):
        d_time = self.start
        single_neuron = self.n_layer_out == 1

        teacher_dicts = {}
        for teacher in self.teacher_layer.get('global_id'):
            teacher_dicts[teacher] = {'amplitude_times': [], 'amplitude_values': []}
        # TODO
        # calc amplitude times one time and concatenate
        for _ in range(self.epochs):
            for spikes, cl in zip(input_spikes, classes):
                current_teacher_id = (
                    self.teacher_layer[0] if single_neuron else self.teacher_layer[cl]
                )
                current_teacher = teacher_dicts[current_teacher_id.get('global_id')]
                start_of_stimulation = d_time + self.reinforce_delta
                end_of_stimulation = start_of_stimulation + self.reinforce_time + self.h
                current_teacher['amplitude_times'].append(start_of_stimulation)
                current_teacher['amplitude_times'].append(end_of_stimulation)

                current_teacher['amplitude_values'].append(self.teacher_amplitude)
                current_teacher['amplitude_values'].append(0.0)
                d_time += self.h_time
        return teacher_dicts


class TeacherFull(Teacher):
    """
    Teacher signal generator for Network class
    """

    def __init__(self, epochs, **kwargs):
        super(TeacherFull, self).__init__(**kwargs)
        self.epochs = epochs

    def create_teacher_dict(
        self, stimulation_start, stimulation_end, classes, teachers, teacher_amplitude
    ):
        single_neuron = self.n_layer_out == 1
        # epochs = settings['learning']['epochs']
        classes_full = np.tile(classes, self.epochs)
        teacher_dict = {}
        for teacher in teachers.get('global_id'):
            teacher_dict[teacher] = {
                'amplitude_times': np.ndarray([]),
                'amplitude_values': np.ndarray([]),
            }
        for cl in np.unique(classes):
            class_mask = classes_full == cl
            stimulation_start_current = stimulation_start[class_mask]
            stimulation_end_current = stimulation_end[class_mask]
            current_teacher_id = teachers[0] if single_neuron else teachers[cl]
            amplitude_times = np.stack(
                (stimulation_start_current, stimulation_end_current), axis=-1
            ).flatten()
            amplitude_values = np.stack(
                (
                    np.full_like(stimulation_start_current, teacher_amplitude),
                    np.zeros_like(stimulation_end_current),
                ),
                axis=-1,
            ).flatten()
            assert len(amplitude_times) == len(stimulation_start_current) * 2
            assert len(amplitude_values) == len(stimulation_end_current) * 2
            teacher_dict[current_teacher_id.get('global_id')]['amplitude_times'] = (
                amplitude_times
            )
            teacher_dict[current_teacher_id.get('global_id')]['amplitude_values'] = (
                amplitude_values
            )
        return teacher_dict


class TeacherInhibitory(Teacher):
    def __init__(self, **kwargs):
        super(TeacherInhibitory, self).__init__(**kwargs)

    def create_teacher_dict(
        self, stimulation_start, stimulation_end, classes, teachers, teacher_amplitude
    ):
        single_neuron = self.n_layer_out == 1
        # epochs = settings['learning']['epochs']
        # classes_full = np.tile(classes, epochs)
        teacher_dict = {}
        for teacher in teachers.get('global_id'):
            teacher_dict[teacher] = {
                'amplitude_times': np.ndarray([]),
                'amplitude_values': np.ndarray([]),
            }
        for cl in np.unique(classes):
            current_teacher_id = teachers[0] if single_neuron else teachers[cl]
            class_mask = classes == cl
            stimulation_start_current = stimulation_start[class_mask]
            stimulation_end_current = stimulation_end[class_mask]
            amplitude_times = np.stack(
                (stimulation_start_current, stimulation_end_current), axis=-1
            ).flatten()
            amplitude_values_pos = np.stack(
                (
                    np.full_like(stimulation_start_current, teacher_amplitude),
                    np.zeros_like(stimulation_end_current),
                ),
                axis=-1,
            ).flatten()
            amplitude_values_neg = np.stack(
                (
                    np.full_like(stimulation_start_current, -teacher_amplitude),
                    np.zeros_like(stimulation_end_current),
                ),
                axis=-1,
            ).flatten()
            assert len(amplitude_times) == len(stimulation_start[class_mask]) * 2
            assert len(amplitude_values_pos) == len(stimulation_start[class_mask]) * 2
            assert len(amplitude_values_neg) == len(stimulation_start[class_mask]) * 2
            for teacher_id in teachers:
                if current_teacher_id != teacher_id:
                    teacher_dict[teacher_id.get('global_id')]['amplitude_times'] = (
                        amplitude_times
                    )
                    teacher_dict[teacher_id.get('global_id')]['amplitude_values'] = (
                        amplitude_values_neg
                    )
                # else:
                #     teacher_dict[teacher_id.get('global_id')]['amplitude_values'] = amplitude_values_pos
        return teacher_dict


class TeacherInhibitoryFull(Teacher):
    def __init__(self, epochs, **kwargs):
        super(TeacherInhibitoryFull, self).__init__(**kwargs)
        self.epochs = epochs

    def create_teacher_dict(
        self, stimulation_start, stimulation_end, classes, teachers, teacher_amplitude
    ):
        single_neuron = self.n_layer_out == 1
        classes_full = np.tile(classes, self.epochs)
        teacher_dict = {}
        for teacher in teachers.get('global_id'):
            teacher_dict[teacher] = {
                'amplitude_times': np.ndarray([]),
                'amplitude_values': np.ndarray([]),
            }
        for cl in np.unique(classes):
            current_teacher_id = teachers[0] if single_neuron else teachers[cl]
            class_mask = classes_full == cl
            stimulation_start_current = stimulation_start[class_mask]
            stimulation_end_current = stimulation_end[class_mask]
            amplitude_times = np.stack(
                (stimulation_start_current, stimulation_end_current), axis=-1
            ).flatten()
            amplitude_values_pos = np.stack(
                (
                    np.full_like(stimulation_start_current, teacher_amplitude),
                    np.zeros_like(stimulation_end_current),
                ),
                axis=-1,
            ).flatten()
            amplitude_values_neg = np.stack(
                (
                    np.full_like(stimulation_start_current, -teacher_amplitude),
                    np.zeros_like(stimulation_end_current),
                ),
                axis=-1,
            ).flatten()
            assert len(amplitude_times) == len(stimulation_start[class_mask]) * 2
            assert len(amplitude_values_pos) == len(stimulation_start[class_mask]) * 2
            assert len(amplitude_values_neg) == len(stimulation_start[class_mask]) * 2
            for teacher_id in teachers:
                if current_teacher_id != teacher_id:
                    teacher_dict[teacher_id.get('global_id')]['amplitude_times'] = (
                        amplitude_times
                    )
                    teacher_dict[teacher_id.get('global_id')]['amplitude_values'] = (
                        amplitude_values_neg
                    )
                # else:
                #     teacher_dict[teacher_id.get('global_id')]['amplitude_values'] = amplitude_values_pos
        return teacher_dict


class ReinforceTeacher(Teacher):
    def __init__(self, **kwargs):
        super(ReinforceTeacher, self).__init__(**kwargs)

    def create_teacher(self, input_spikes, classes):
        full_time = len(input_spikes) * self.h_time  # + start
        times = np.arange(0, full_time, self.h_time)
        pattern_start_times = np.expand_dims(
            np.tile(times, (len(input_spikes[0]), 1)).T, axis=2
        )
        assert len(input_spikes) == len(pattern_start_times)

        spike_times = np.add(input_spikes, pattern_start_times)
        stimulation_start = np.nanmin(spike_times, axis=1) + self.reinforce_delta
        stimulation_end = stimulation_start + self.reinforce_time + 2 * self.h
        assert len(stimulation_start) == len(spike_times)

        teacher_dict = self.create_teacher_dict(
            stimulation_start,
            stimulation_end,
            classes,
            self.teacher_layer,
            self.teacher_amplitude,
        )
        return teacher_dict

    def create_teacher_dict(
        self, stimulation_start, stimulation_end, classes, teachers, teacher_amplitude
    ):
        single_neuron = self.n_layer_out == 1
        teacher_dict = {}
        for teacher in teachers.get('global_id'):
            teacher_dict[teacher] = {
                'amplitude_times': np.ndarray([]),
                'amplitude_values': np.ndarray([]),
            }

        for cl in range(len(teachers)):
            class_mask = classes == cl

            stimulation_start_current = stimulation_start[class_mask]
            stimulation_end_current = stimulation_end[class_mask]
            current_teacher_id = teachers[0] if single_neuron else teachers[cl]

            amplitude_times = np.stack(
                (stimulation_start_current, stimulation_end_current), axis=-1
            ).flatten()
            amplitude_values = np.stack(
                (
                    np.full_like(stimulation_start_current, teacher_amplitude),
                    np.zeros_like(stimulation_end_current),
                ),
                axis=-1,
            ).flatten()
            assert len(amplitude_times) == len(stimulation_start_current) * 2
            assert len(amplitude_values) == len(stimulation_end_current) * 2
            teacher_dict[current_teacher_id.get('global_id')]['amplitude_times'] = (
                amplitude_times
            )
            teacher_dict[current_teacher_id.get('global_id')]['amplitude_values'] = (
                amplitude_values
            )
        return teacher_dict
