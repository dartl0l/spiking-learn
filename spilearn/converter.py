# coding: utf-8

import numpy as np

from .andrews_curve import AndrewsCurve


class Converter:
    def __init__(self):
        pass

    def convert(self, x, y):
        output = {'input': x,
                  'class': y}
        return output
    
    def convert_data_to_patterns(self, x, y, pattern_time, h):
        """
            Function must be updated to O(n) complexity 
        """
        input_for_nn = []
        #     h = 0.1
        #     time = 60
        rad = 2 * np.pi
        # rad_to_deg = 57

        for xx, yy in zip(x, y):
            pattern = {}
            curve = AndrewsCurve(xx, rad, h)
            #         pattern['times'] = np.round(curve.theta() * rad_to_deg, 1)
            pattern['times'] = np.round(curve.theta() * (pattern_time / rad), 1)

            pattern['neurons'] = map(int, curve.discrete_curve())
            pattern['class'] = int(yy)
            input_for_nn.append(pattern)

        min_neuron = 0

        for pattern in input_for_nn:
            if np.amin(pattern['neurons']) < min_neuron:
                min_neuron = np.amin(pattern['neurons'])

        for pattern in input_for_nn:
            pattern['neurons'] += np.abs(min_neuron)

        output = []
        for pattern in input_for_nn:
            tmp_dic = {}
            input_dic = dict.fromkeys(pattern['neurons'])

            for key in input_dic:
                input_dic[key] = []

            for neuron, time in zip(pattern['neurons'], pattern['times']):
                input_dic[neuron].append(time)

            tmp_dic['input'] = input_dic
            tmp_dic['class'] = pattern['class']

            output.append(tmp_dic)
        return output

    def convert_data_to_patterns_spatio_temp(self, x, y, min_time, h):
        """
            Function must be updated to O(n) complexity 
        """
        rad = 2 * np.pi
        # rad_to_deg = 57

        output = []
        for xx, yy in zip(x, y):
            pattern = {}

            curve = AndrewsCurve(xx, rad, h).curve()

            tmp_dict = dict.fromkeys(np.arange(0, int(rad / h)))

            for i in range(int(rad / h)):
                tmp_dict[i] = np.round([(curve[i] + min_time) * 20], 1)

            pattern['input'] = tmp_dict
            pattern['class'] = yy

            output.append(pattern)
        return output

    def convert_data_to_patterns_uniform(self, x, y, pattern_time, h, mult):
        """
            Function must be updated to O(n) complexity 
        """
        output = []

        for xx, yy in zip(x, y):
            pattern = {}

            tmp_dict = dict.fromkeys(np.arange(0, len(xx) * mult))

            i = 0

            for x in xx:
                time = x * pattern_time

                for _ in range(mult):
                    tmp_dict[i] = np.sort(np.random.uniform(0, int(time), int(time / h)) + 0.1)
                    # get_poisson_train(time, firing_rate, h)
                    # tmp_dict[i] = get_poisson_train(time, firing_rate, h)
                    i += 1

            pattern['input'] = tmp_dict
            pattern['class'] = yy

            output.append(pattern)
        return output

    def convert_data_to_patterns_gaussian_receptive_field(self, x, y, sigma2, max_x,
                                                          n_fields, k_round, duplicate=False):
        """
            Function must be updated to O(n) complexity 
        """
        def get_gaussian(x, sigma2, mu):
            return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma2))

        output = {'input': [],
                  'class': []}

        h_mu = max_x / (n_fields - 1)

        max_y = np.round(get_gaussian(h_mu, sigma2, h_mu), 0)

        for xx, yy in zip(x, y):
            tmp_dict = dict.fromkeys(np.arange(0, n_fields * len(xx)))

            tmp_fields = 0
            for x in xx:
                mu = 0
                for i in range(n_fields):
                    time = np.round(get_gaussian(x, sigma2, mu), k_round)
                    spike_time = max_y - time
                    if duplicate:
                        tmp_dict[i + tmp_fields] = [spike_time, spike_time + max_y / 2]
                    else:
                        tmp_dict[i + tmp_fields] = [spike_time]
                    mu += h_mu
                tmp_fields += n_fields
            output['input'].append(tmp_dict)
            output['class'].append(yy)
        output = {'input': np.array(output['input']),
                  'class': np.array(output['class'])}
        return output, max_y
    
    def convert_image_to_spikes_without_last(self, x, y, pattern_length, k_round):
        """
            Function must be updated to O(n) complexity 
        """
        zero_values = x == 0
        
        X = pattern_length * (1 - x)
        X[zero_values] = 0
        
        output = {'input': [],
                  'class': []}
        for xx, yy in zip(X, y):
            tmp_dict = dict.fromkeys(np.arange(0, len(xx)))
            for i, x in enumerate(xx):
                time = np.round(x, k_round)
                tmp_dict[i] = [time] if time != 0 else []
                    
            output['input'].append(tmp_dict)
            output['class'].append(yy)
        output = {'input': np.array(output['input']),
                  'class': np.array(output['class'])}
        return output
    
    def convert_data_to_patterns_poisson(self, x, y, pattern_time, firing_rate, h):
        """
            Function must be updated to O(n) complexity 
        """
        def get_poisson_train(time, firing_rate, h):
            np.random.seed()
            dt = 1.0 / 1000.0 * h
            times = np.arange(0, time, h)
            mask = np.random.random(int(time / h)) < firing_rate * dt
            spike_times = times[mask]
            return spike_times

        output = {'input': [],
                  'class': []}

        for xx, yy in zip(x, y):
            tmp_dict = dict.fromkeys(np.arange(0, len(xx)))
            for i, x in enumerate(xx):
                time = x * pattern_time
                tmp_dict[i] = get_poisson_train(time, firing_rate, h)
            output['input'].append(tmp_dict)
            output['class'].append(yy)
        output = {'input': np.array(output['input']),
                  'class': np.array(output['class'])}
        return output


class ReceptiveFieldsConverter(Converter):
    """
        Class for receptive fields data conversion
    """
    def __init__(self, sigma2, max_x, n_fields, k_round, scale=1.0,
                 reshape=True, reverse=False, no_last=False):
        self.sigma2 = sigma2
        self.max_x = max_x
        self.n_fields = n_fields
        self.k_round = k_round
        self.scale = scale
        self.reshape = reshape
        self.reverse = reverse
        self.no_last = no_last

    def get_gaussian(self, x, sigma2, mu):
        return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma2))

    def convert(self, x, y):
        """
            Function must be updated to O(n) complexity 
        """

        h_mu = self.max_x / (self.n_fields - 1)

        max_y = np.round(self.get_gaussian(h_mu, self.sigma2, h_mu), 0)

        mu = np.tile(np.linspace(0, self.max_x, self.n_fields), len(x[0]))
        x = np.repeat(x, self.n_fields, axis=1)

        assert len(mu) == len(x[0])

        if self.reverse:
            x = np.round(self.get_gaussian(x, self.sigma2, mu), self.k_round)
            if self.no_last:
                mask = x < 0.1
                x[mask] = np.nan
        else:
            x = max_y - np.round(self.get_gaussian(x, self.sigma2, mu), self.k_round)
            if self.no_last:
                mask = x > max_y - 0.09
                x[mask] = np.nan

        x *= self.scale
        if self.reshape:
            output = {'input': x.reshape(x.shape[0], x.shape[1], 1),
                      'class': y}
        else:
            output = {'input': x,
                      'class': y}
        return output  # , max_y


# class ReceptiveFieldsConverterWithoutLast(Converter):
#     """
#         Class for receptive fields data conversion
#     """
#     def __init__(self, sigma2, max_x, n_fields, k_round, reshape=True):
#         self.sigma2 = sigma2
#         self.max_x = max_x
#         self.n_fields = n_fields
#         self.k_round = k_round
#         self.reshape = reshape
#
#     def get_gaussian(self, x, sigma2, mu):
#         return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma2))
#
#     def convert(self, x, y):
#         """
#             Function must be updated to O(n) complexity
#         """
#
#         h_mu = self.max_x / (self.n_fields - 1)
#
#         max_y = np.round(self.get_gaussian(h_mu, self.sigma2, h_mu), 0)
#
#         mu = np.tile(np.linspace(0, self.max_x, self.n_fields), len(x[0]))
#         x = np.repeat(x, self.n_fields, axis=1)
#
#         assert len(mu) == len(x[0])
#
#         x = max_y - np.round(self.get_gaussian(x, self.sigma2, mu), self.k_round)
#         mask = x[x > max_y - 0.09]
#         x[mask] = np.nan
#         if self.reshape:
#             output = {'input': x.reshape(x.shape[0], x.shape[1], 1),
#                       'class': y}
#         else:
#             output = {'input': x,
#                       'class': y}
#         return output  # , max_y


# class ReceptiveFieldsConverterReverse(ReceptiveFieldsConverter):
#     """
#         Class for receptive fields data conversion
#     """
#     def __init__(self, sigma2, max_x, n_fields, k_round, reshape=True):
#         self.sigma2 = sigma2
#         self.max_x = max_x
#         self.n_fields = n_fields
#         self.k_round = k_round
#         self.reshape = reshape
#
#     def convert(self, x, y):
#         """
#             Function must be updated to O(n) complexity
#         """
#
#         output = {'input': [],
#                   'class': []}
#
#         h_mu = self.max_x / (self.n_fields - 1)
#
#         # max_y = np.round(get_gaussian(h_mu, self.sigma2, h_mu), 0)
#
#         mu = np.tile(np.linspace(0, self.max_x, self.n_fields), len(x[0]))
#         x = np.repeat(x, self.n_fields, axis=1)
#
#         assert len(mu) == len(x[0])
#
#         x = np.round(self.get_gaussian(x, self.sigma2, mu), self.k_round)
#         if self.reshape:
#             output = {'input': x.reshape(x.shape[0], x.shape[1], 1),
#                       'class': y}
#         else:
#             output = {'input': x,
#                       'class': y}
#         return output  # , max_y


class TemporalConverter(Converter):
    '''
        Class for receptive fields data conversion
    '''
    def __init__(self, pattern_length, k_round, reshape=True, reverse=False, no_last=False):
        self.pattern_length = pattern_length
        self.k_round = k_round
        self.reshape = reshape
        self.reverse = reverse
        self.no_last = no_last

    def convert(self, x, y):
        '''
            Function must be updated to O(n) complexity 
        '''

        if self.no_last:
            zero_values = x == 0
            x[zero_values] = np.nan
        if self.reverse:
            x = np.round(self.pattern_length * x, self.k_round)
        else:
            x = np.round(self.pattern_length * (1 - x), self.k_round)

        if self.reshape:
            output = {'input': x.reshape(x.shape[0], x.shape[1], 1),
                      'class': y}
        else:
            output = {'input': x,
                      'class': y}

        return output


# class TemporalInvertedConverter(Converter):
#     '''
#         Class for receptive fields data conversion
#     '''
#     def __init__(self, pattern_length, k_round, reshape=True):
#         self.pattern_length = pattern_length
#         self.k_round = k_round
#         self.reshape = reshape
#
#     def convert(self, x, y):
#         '''
#             Function must be updated to O(n) complexity
#         '''
#
#         x = np.round(self.pattern_length * x, self.k_round)
#         if self.reshape:
#             output = {'input': x.reshape(x.shape[0], x.shape[1], 1),
#                       'class': y}
#         else:
#             output = {'input': x,
#                       'class': y}
#
#         return output
#

class BinaryConverter(Converter):
    '''
        Class for receptive fields data conversion
    '''
    def __init__(self, pattern_length, k_round):
        self.pattern_length = pattern_length
        self.k_round = k_round

    def convert(self, x, y):
        x[x != 0] = 1
        x = np.round(self.pattern_length * (1 - x), self.k_round)
        output = {'input': x.reshape(x.shape[0], x.shape[1], 1),
                  'class': y}

        return output


class TemporalConverterWithoutZero(TemporalConverter):
    '''
        Class for receptive fields data conversion
    '''
    def __init__(self, pattern_length, k_round, reshape=True):
        self.pattern_length = pattern_length
        self.k_round = k_round
        self.reshape = reshape
    
    def convert(self, x, y):
        zero_values = x == 0
        x[zero_values] = np.nan

        x = np.round(self.pattern_length * (1 - x), self.k_round)
        output = {'input': x.reshape(x.shape[0], x.shape[1], 1),
                  'class': y}
        return output

    
class FrequencyConverter(Converter):
    '''
        Class for receptive fields data conversion
    '''
    def __init__(self, pattern_time, firing_rate, dt, h):
        self.pattern_time = pattern_time
        self.firing_rate = firing_rate
        self.h = h
        self.dt = dt

    def convert(self, x, y):
        '''
            Function must be updated to O(n) complexity 
        '''
        def get_poisson_train(time, firing_rate, h):
            np.random.seed()
            times = np.arange(0, time, h)
            probabilities = np.random.uniform(0, 1, int(time / h))
            mask = probabilities < firing_rate
            spike_times = times[mask]
            return spike_times

        output = {'input': [],
                  'class': []}

        for xx, yy in zip(x, y):
            tmp_dict = dict.fromkeys(np.arange(0, len(xx)))
            for i, x in enumerate(xx):
                frequency = x * self.firing_rate * self.dt
                tmp_dict[i] = get_poisson_train(self.pattern_time, frequency, self.h)
            output['input'].append(tmp_dict)
            output['class'].append(yy)
        output = {'input': np.array(output['input']),
                  'class': np.array(output['class'])}
        return output


class SobelConverter(Converter):
    '''
        Class for receptive fields data conversion
    '''
    def __init__(self, pattern_time, firing_rate, dt, h):
        self.pattern_time = pattern_time
        self.firing_rate = firing_rate
        self.h = h
        self.dt = dt

    def convert(self, x, y):
        '''
            Function must be updated to O(n) complexity 
        '''
        def get_sobel_image(image):
            import scipy
            from scipy import ndimage

            # im = im.astype('int32')
            dx = ndimage.sobel(im, 0)  # horizontal derivative
            dy = ndimage.sobel(im, 1)  # vertical derivative
            mag = np.hypot(dx, dy)  # magnitude
            mag *= 255.0 / np.max(mag)  # normalize (Q&D)
            return mag


        output = {'input': [],
                  'class': []}

        for xx, yy in zip(x, y):
            tmp_dict = dict.fromkeys(np.arange(0, len(xx)))
            for i, x in enumerate(xx):
                frequency = x * self.firing_rate * self.dt
                # tmp_dict[i] = get_poisson_train(self.pattern_time, frequency, self.h)
            output['input'].append(tmp_dict)
            output['class'].append(yy)
        output = {'input': np.array(output['input']),
                  'class': np.array(output['class'])}
        return output
