# coding: utf-8

import numpy as np
from math import cos, sin, pi, ceil, sqrt

import matplotlib.pyplot as plt
from matplotlib import animation, rc, cm
from mpl_toolkits.mplot3d import Axes3D


class Plotter:
    def __init__(self, grayscale=False):
        if grayscale:
            plt.style.use('grayscale')

    def plot_field(self, sigma2, max_x, n_fields):

        def get_gaussian(x, sigma2, mu):
            return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma2))

        h_mu = max_x / (n_fields - 1)
        mu = 0
        for _ in range(n_fields):
            xx = np.arange(0, max_x, 0.01)
            yy = [get_gaussian(j, sigma2, mu) for j in xx]
            plt.plot(xx, yy)
            # left += h_mu
            # right += h_mu
            #         print mu
            mu += h_mu
        plt.xlabel('Value $x$ of the input vector component')
        plt.ylabel('Spike times of pattern')
        plt.show()

    def plot_weights_2d(self, weights, rows, columns, show=True):
        plt.clf()
        # plt.figure()
        # plt.title('Weight distribution')

        neurons = weights.keys()
        num_neurons = len(weights.keys())
        ax_rows = int(ceil(sqrt(num_neurons)))
        ax_cols = int(ceil(num_neurons / ax_rows))
        fig, axs = plt.subplots(nrows=ax_rows, ncols=ax_cols, figsize=(20, 20))

        for ax, neuron in zip(axs.flat, neurons):
            current_weights = np.array(weights[neuron]).reshape((rows, columns))
            ax.matshow(current_weights)
            ax.set_title('Neuron ' + str(neuron) + ' weights')

        plt.tight_layout()
        if show:
            plt.show()

    def plot_weights(self, weights, show=True):
        plt.clf()
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        plt.title('Weight distribution')
        for neuron in weights:
            plt.plot(list(range(len(weights[neuron]))),
                     weights[neuron], '.',
                     label=str(neuron))
        plt.xlabel('Input synapse number')
        plt.ylabel('Synapse weight')
        plt.legend()
        if show:
            plt.show()

    def plot_weights_rus(self, weights, show=True):
        plt.clf()
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        plt.title('Распределение весов')
        for neuron in weights:
            plt.plot(list(range(len(weights[neuron]))),
                     weights[neuron], '.',
                     label='Нейрон ' + str(neuron))
        plt.xlabel('Номер входного синапса')
        plt.ylabel('Синаптический вес')
        plt.legend()
        if show:
            plt.show()
            
    def plot_norm(self, norm_history, show=True):
        plt.clf()
        plt.title('Weight norm')
        # norm_history = np.array(norm_history).T.tolist()
        # print(norm_history)
        plt.plot(list(range(len(norm_history))),
                 norm_history, '-')
        # plt.legend()
        if show:
            plt.show()

    def plot_norms(self, norm_history, show=True):
        plt.clf()
        plt.title('Weight norms')
        colors = ['-r', '-g', '-b', '-y']
        # norm_history = np.array(norm_history).T.tolist()
        # print(norm_history)
        for i, norms in enumerate(norm_history):
            plt.plot(list(range(len(norms.tolist()))),
                     norms, colors[i], label=str(i))
        plt.legend()
        if show:
            plt.show()

    def plot_animated_weights(self, weights_history, h, save, show):
        def plot_weights_for_anim(weights):
            neurons = weights.keys()
            plt.title('Weight distribution')

            plot = plt.plot(range(len(weights[neurons[0]])), weights[neurons[0]], 'r.',
                            range(len(weights[neurons[1]])), weights[neurons[1]], 'b.')

            return plot

        all_plot = []
        fig = plt.figure()

        for i in range(0, len(weights_history), h):
            weights = weights_history[i]
            all_plot.append(plot_weights_for_anim(weights['layer_0']))

        weights_anim = animation.ArtistAnimation(fig, all_plot, blit=True)

        if save:
            weights_anim.save('weights.mp4')
            # if show is True:
            #     HTML(weights_anim.to_html5_video())
        return weights_anim

    def plot_voltage(self, voltmeter, legend=True):
        plt.clf()
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        neurons = set(voltmeter['senders'])
        assert len(voltmeter['senders']) == len(voltmeter['V_m']) == len(voltmeter['times'])

        plt.title("Membrane potential")
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane potential (mV)')
        for neuron in neurons:
            mask = voltmeter['senders'] == neuron
            assert len(voltmeter['times'][mask]) == len(voltmeter['V_m'][mask])
            plt.plot(voltmeter['times'][mask],
                     voltmeter['V_m'][mask],
                     label='neuron ' + str(neuron))
        if legend:
            plt.legend()

    def plot_spikes(self, spike_detector):
        plt.clf()
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        neurons = set(spike_detector['senders'])
        assert len(spike_detector['senders'])  == len(spike_detector['times'])

        plt.title("Spikes")
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron')
        for neuron in neurons:
            mask = spike_detector['senders'] == neuron
            assert len(spike_detector['times'][mask]) == len(spike_detector['senders'][mask])
            plt.plot(spike_detector['times'][mask],
                     spike_detector['senders'][mask], 'b.')
        # plt.legend()

    def plot_devices(self, devices, plot_last_detector=False):
        self.plot_voltage(devices['voltmeter'])
        plt.show()
        
        if plot_last_detector:
            self.plot_voltage(devices['voltmeter_hidden'], False)
            plt.show()

            self.plot_spikes(devices['spike_detector_hidden'])
            plt.show()

        self.plot_spikes(devices['spike_detector_input'])
        plt.show()

        self.plot_spikes(devices['spike_detector_out'])
        plt.show()

    def plot_devices_limits(self, devices, start, end, plot_last_detector=False):
        self.plot_voltage(devices['voltmeter'])
        plt.xlim(start, end)
        plt.show()

        if plot_last_detector:
            self.plot_voltage(devices['voltmeter_hidden'])
            plt.xlim(start, end)
            plt.show()

            self.plot_spikes(devices['spike_detector_hidden'])
            plt.xlim(start, end)
            plt.show()

        self.plot_spikes(devices['spike_detector_input'])
        plt.xlim(start, end)
        plt.show()

        self.plot_spikes(devices['spike_detector_out'])
        plt.xlim(start, end)
        plt.show()

    def plot_latency(self, latency, classes, title, show=True):
        plt.clf()
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron')
        
        plt.title(title)
        # colors = ['rx', 'gx', 'bx', 'cx', 'mx', 'yx', 'kx',
        #           'ro', 'go', 'bo', 'co', 'mo', 'yo', 'ko']
#         shapes = ['x', 's', 'd']
        
        classes_set = set(classes)
        for cl in classes_set:
            class_mask = classes == cl
            latencies = np.array(latency)[class_mask]
            neurons = np.tile(np.arange(len(classes_set)), (len(latencies), 1))
            plt.plot(latencies.flatten(), neurons.flatten(), '.',
                     # colors[cl],
                     label='class ' + str(cl))
        plt.legend()
        if show:
            plt.show()

    def plot_latencies(self, latencies, title, show=True):
        plt.title(title)
        for latency in latencies:
            if list(latency['latency']):
                if latency['class'] == 0:
                    plt.plot(latency['latency'][:1], 1, 'rx')
                elif latency['class'] == 1:
                    plt.plot(latency['latency'][:1], 2, 'gx')
                elif latency['class'] == 2:
                    plt.plot(latency['latency'][:1], 3, 'bx')
                elif latency['class'] == 3:
                    plt.plot(latency['latency'][:1], 4, 'bx')
                elif latency['class'] == 4:
                    plt.plot(latency['latency'][:1], 5, 'bx')
                elif latency['class'] == 5:
                    plt.plot(latency['latency'][:1], 6, 'bx')
                elif latency['class'] == 6:
                    plt.plot(latency['latency'][:1], 7, 'bx')
                elif latency['class'] == 7:
                    plt.plot(latency['latency'][:1], 8, 'bx')
                elif latency['class'] == 8:
                    plt.plot(latency['latency'][:1], 9, 'bx')
                elif latency['class'] == 9:
                    plt.plot(latency['latency'][:1], 10, 'bx')
        if show:
            plt.show()

    def plot_train_latency(self, latency_train, title, show=True):
        latency_paint = {'latency': [],
                         'epoch': []}

        epoch = 1

        for latency in latency_train:
            if list(latency['latency']):
                for lat in latency['latency']:
                    latency_paint['latency'].append(lat)
                    latency_paint['epoch'].append(epoch)
            else:
                latency_paint['latency'].append('nan')
                latency_paint['epoch'].append(epoch)
            epoch += 1

        plt.plot(latency_paint['epoch'], latency_paint['latency'], 'b.')
        plt.xlabel('Epochs')
        plt.ylabel('Latency')
        plt.title(title)
        if show:
            plt.show()

    def plot_pattern(self, pattern, show=True):
        plt.clf()
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        plt.ylim(0, 30)
        plt.xlim(0, len(pattern))
        plt.title('Temporal pattern')
        for neuron in range(len(pattern)):
            # if pattern[neuron]:
            plt.plot(neuron, pattern[neuron], 'b.')
        if show:
            plt.show()

    def plot_frequency_pattern(self, pattern, time, show=True):
        plt.ylim(0, len(pattern))
        plt.xlim(0, time)
        plt.title('Frequency pattern')
        for neuron in range(len(pattern)):
            plt.plot(pattern[neuron], [neuron] * len(pattern[neuron]), 'b.')
        if show:
            plt.show()

    def plot_image(self, image_spikes, image_size, title, show=True):
        # plt.xlim(0, image_size[0])
        # plt.ylim(0, image_size[1])

        scale = 100
        plt.title(title)

        spike_pos_x = 0
        spike_pos_y = image_size[1]
        new_image = []
        for neuron in image_spikes.keys():
            if image_spikes[neuron]:
                new_image.append(image_spikes[neuron][0])
            else:
                new_image.append(0)
        new_image = np.array(new_image).reshape(image_size)

        x = np.arange(0, image_size[0], 1)
        y = np.arange(0, image_size[1], 1)
        x, y = np.meshgrid(x, y)

        plt.scatter(x, y, new_image * scale)

        if show:
            plt.show()

    def plot_params(self, parameters_acc_pairs,
                    title='Parameters distribution',
                    normalize_colors=True, show=True):
        def get_polar(r, fi):
            return r * cos(fi), r * sin(fi)

        def get_polars(parameters):
            x_list = []
            y_list = []
            h_fi = 2 * pi / len(parameters)

            for i, parameter in enumerate(parameters):
                x, y = get_polar(parameter, h_fi * i)
                x_list.append(x)
                y_list.append(y)
            x_list.append(x_list[0])
            y_list.append(y_list[0])
            return x_list, y_list

        def get_axes(n_axes):
            h_fi = 2 * pi / n_axes

            axes = []

            for i in range(n_axes):
                x, y = get_polar(1.0, h_fi * i)
                x_list = [0, x]
                y_list = [0, y]
                axes.append((x_list, y_list))
            return axes

        accs = []
        all_params = []
        param_names = sorted(list(parameters_acc_pairs[0][0].keys()))

        for parameters, acc in parameters_acc_pairs:
            accs.append(acc)
            all_params.append([x for _, x in sorted(zip(list(parameters.keys()), list(parameters.values())))])

        plt.title(title)

        plt.ylim(-1.5, 1.5)
        plt.xlim(-1.5, 1.5)

        axes = get_axes(len(param_names))

        for (x, y), param in zip(axes, param_names):
            plt.plot(x, y, 'k-')
            plt.text(x[1], y[1], param)

        cmap = cm.get_cmap('viridis')
        for acc, parameters in sorted(zip(accs, all_params)):
            x, y = get_polars(parameters)
            if normalize_colors:
                color = cmap((acc - 0.8) / 0.2)
            else:
                color = cmap(acc)
            plt.plot(x, y, '-', color=color)
        if show:
            plt.show()

    def plot_pca(self, X, y, show=True):
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        Xpca = pca.fit_transform(X)

        plt.clf()
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        
        # color_shape = ('r.', 'b.', 'g.', 'c.', 'k.', 'm.', 'y.', 'rx', 'bx', 'gx')
        classes_set = set(y)
        for cl in classes_set:
            class_mask = y == cl
            plt.plot(Xpca[class_mask].T[0], Xpca[class_mask].T[1], '.',
                     label='class ' + str(cl))
        plt.legend()
        if show:
            plt.show()
            
    def plot_latency_pca(self, x, y, max_val, show=True):
        from sklearn.decomposition import PCA

        latency_list = np.nan_to_num(x, nan=max_val)
        pca = PCA(n_components=2)
        Xpca = pca.fit_transform(latency_list)

        plt.clf()
        fig = plt.figure()
        fig.patch.set_facecolor('white')

        # color_shape = ('r.', 'b.', 'g.', 'c.', 'k.', 'm.', 'y.', 'rx', 'bx', 'gx')
        classes_set = set(y)
        for cl in classes_set:
            class_mask = y == cl
            plt.plot(Xpca[class_mask].T[0], Xpca[class_mask].T[1], '.',
                     label='class ' + str(cl))
        plt.legend()
        if show:
            plt.show()