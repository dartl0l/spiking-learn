import numpy as np


class Converter:
    def __init__(self):
        pass

    def convert_data_to_patterns(self, x, y, pattern_time, h):
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
                                                          n_fields, k_round):
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
                    # time = get_gaussian(x, sigma2, mu)
                    # if time > 0.09:
                    #     tmp_dict[i + tmp_fields] = [k_time * (max_y - time)]
                    # else:
                    #     tmp_dict[i + tmp_fields] = []

                    tmp_dict[i + tmp_fields] = [max_y - time]

                    mu += h_mu
                tmp_fields += n_fields
            output['input'].append(tmp_dict)
            output['class'].append(yy)
        output = {'input': np.array(output['input']),
                  'class': np.array(output['class'])}
        return output, max_y

    def convert_image_to_patterns_gaussian_receptive_field(self, x, y, sigma2, k_round):
        def get_gaussian(x, sigma2, mu):
            return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma2))

        output = {'input': [],
                  'class': []}

        mu = 1.0
        # max_y = np.round(get_gaussian(mu, sigma2, mu), 0)
        for xx, yy in zip(x, y):
            tmp_dict = dict.fromkeys(np.arange(0, len(xx)))
            for i, x in enumerate(xx):
                time = np.round(get_gaussian(x, sigma2, mu), k_round)
                # print(time)
                tmp_dict[i] = [time]
            output['input'].append(tmp_dict)
            output['class'].append(yy)
        output = {'input': np.array(output['input']),
                  'class': np.array(output['class'])}
        return output  # , max_y

    def convert_data_to_patterns_poisson(self, x, y, pattern_time, firing_rate, h, mult):
        def get_poisson_train(time, firing_rate, h):
            np.random.seed()
            dt = 1.0 / 1000.0 * h
            times = np.arange(0, time, h)
            mask = np.random.random(int(time / h)) < firing_rate * dt
            spike_times = times[mask]
            return spike_times

        output = []

        for xx, yy in zip(x, y):
            pattern = {}

            tmp_dict = dict.fromkeys(np.arange(0, len(xx) * mult))

            i = 0

            for x in xx:
                time = x * pattern_time

                for _ in range(mult):
                    tmp_dict[i] = get_poisson_train(time, firing_rate, h)
                    i += 1

            pattern['input'] = tmp_dict
            pattern['class'] = yy

            output.append(pattern)
        return output
