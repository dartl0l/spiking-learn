# coding: utf-8

import pylab as pl
import numpy as np
from math import cos, sin, pi
from matplotlib import animation, rc, cm
from mpl_toolkits.mplot3d import Axes3D


class Plotter:
    def __init__(self, grayscale=False):
        if grayscale:
            pl.style.use('grayscale')

    def plot_field(self, sigma2, max_x, n_fields):

        def get_gaussian(x, sigma2, mu):
            return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma2))

        h_mu = max_x / (n_fields - 1)
        mu = 0
        for _ in range(n_fields):
            xx = np.arange(0, max_x, 0.01)
            yy = [get_gaussian(j, sigma2, mu) for j in xx]
            pl.plot(xx, yy)
            # left += h_mu
            # right += h_mu
            #         print mu
            mu += h_mu
        pl.xlabel('Value $x$ of the input vector component')
        pl.ylabel('Spike times of pattern')
        pl.show()

    def plot_weights(self, weights, show=True):
        pl.clf()
        fig = pl.figure()
        fig.patch.set_facecolor('white')
        pl.title('Weight distribution')
        for neuron in weights:
            pl.plot(list(range(len(weights[neuron]))),
                    weights[neuron], '.',
                    label=str(neuron))
        pl.xlabel('Input synapse number')
        pl.ylabel('Synapse weight')
        pl.legend()
        if show:
            pl.show()

    def plot_weights_rus(self, weights, show=True):
        pl.clf()
        fig = pl.figure()
        fig.patch.set_facecolor('white')
        pl.title('Распределение весов')
        for neuron in weights:
            pl.plot(list(range(len(weights[neuron]))),
                    weights[neuron], '.',
                    label='Нейрон ' + str(neuron))
        pl.xlabel('Номер входного синапса')
        pl.ylabel('Синаптический вес')
        pl.legend()
        if show:
            pl.show()
            
    def plot_norm(self, norm_history, show=True):
        pl.clf()
        pl.title('Weight norm')
        # norm_history = np.array(norm_history).T.tolist()
        # print(norm_history)
        pl.plot(list(range(len(norm_history))),
                norm_history, '-')
        # pl.legend()
        if show:
            pl.show()

    def plot_norms(self, norm_history, show=True):
        pl.clf()
        pl.title('Weight norms')
        colors = ['-r', '-g', '-b', '-y']
        # norm_history = np.array(norm_history).T.tolist()
        # print(norm_history)
        for i, norms in enumerate(norm_history):
            pl.plot(list(range(len(norms.tolist()))),
                    norms, colors[i], label=str(i))
        pl.legend()
        if show:
            pl.show()

    def plot_animated_weights(self, weights_history, h, save, show):
        def plot_weights_for_anim(weights):
            neurons = weights.keys()
            pl.title('Weight distribution')

            plot = pl.plot(range(len(weights[neurons[0]])), weights[neurons[0]], 'r.',
                           range(len(weights[neurons[1]])), weights[neurons[1]], 'b.')

            return plot

        all_plot = []
        fig = pl.figure()

        for i in range(0, len(weights_history), h):
            weights = weights_history[i]
            all_plot.append(plot_weights_for_anim(weights['layer_0']))

        weights_anim = animation.ArtistAnimation(fig, all_plot, blit=True)

        if save:
            weights_anim.save('weights.mp4')
        #     if show is True:
        #         HTML(weights_anim.to_html5_video())
        return weights_anim

    def plot_voltage(self, voltmeter, legend=True):
        pl.clf()
        fig = pl.figure()
        fig.patch.set_facecolor('white')
        neurons = set(voltmeter['senders'])
        assert len(voltmeter['senders']) == len(voltmeter['V_m']) == len(voltmeter['times'])

        pl.title("Membrane potential")
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane potential (mV)')
        for neuron in neurons:
            mask = voltmeter['senders'] == neuron
            assert len(voltmeter['times'][mask]) == len(voltmeter['V_m'][mask])
            pl.plot(voltmeter['times'][mask],
                    voltmeter['V_m'][mask],
                    label='neuron ' + str(neuron))
        if legend:
            pl.legend()

    def plot_spikes(self, spike_detector):
        pl.clf()
        fig = pl.figure()
        fig.patch.set_facecolor('white')
        neurons = set(spike_detector['senders'])
        assert len(spike_detector['senders'])  == len(spike_detector['times'])

        pl.title("Spikes")
        pl.xlabel('Time (ms)')
        pl.ylabel('Neuron')
        for neuron in neurons:
            mask = spike_detector['senders'] == neuron
            assert len(spike_detector['times'][mask]) == len(spike_detector['senders'][mask])
            pl.plot(spike_detector['times'][mask],
                    spike_detector['senders'][mask], 'b.')
        # pl.legend()

    def plot_devices(self, devices, plot_last_detector=False):
        self.plot_voltage(devices['voltmeter'])
        pl.show()
        
        if plot_last_detector:
            self.plot_voltage(devices['voltmeter_hidden'], False)
            pl.show()

            self.plot_spikes(devices['spike_detector_hidden'])
            pl.show()

        self.plot_spikes(devices['spike_detector_input'])
        pl.show()

        self.plot_spikes(devices['spike_detector_out'])
        pl.show()

    def plot_devices_limits(self, devices, start, end, plot_last_detector=False):
        self.plot_voltage(devices['voltmeter'])
        pl.xlim(start, end)
        pl.show()

        if plot_last_detector:
            self.plot_voltage(devices['voltmeter_hidden'])
            pl.xlim(start, end)
            pl.show()

            self.plot_spikes(devices['spike_detector_hidden'])
            pl.xlim(start, end)
            pl.show()

        self.plot_spikes(devices['spike_detector_input'])
        pl.xlim(start, end)
        pl.show()

        self.plot_spikes(devices['spike_detector_out'])
        pl.xlim(start, end)
        pl.show()

    def plot_latency(self, latency, classes, title, show=True):
        pl.clf()
        fig = pl.figure()
        fig.patch.set_facecolor('white')
        
        pl.xlabel('Time (ms)')
        pl.ylabel('Neuron')
        
        pl.title(title)
        # colors = ['rx', 'gx', 'bx', 'cx', 'mx', 'yx', 'kx',
        #           'ro', 'go', 'bo', 'co', 'mo', 'yo', 'ko']
#         shapes = ['x', 's', 'd']
        
        classes_set = set(classes)
        for cl in classes_set:
            class_mask = classes == cl
            latencies = np.array(latency)[class_mask]
            neurons = np.tile(np.arange(len(classes_set)), (len(latencies), 1))
            pl.plot(latencies.flatten(), neurons.flatten(), '.',
                    # colors[cl],
                    label='class ' + str(cl))
        pl.legend()
        if show:
            pl.show()

    def plot_latencies(self, latencies, title, show=True):
        pl.title(title)
        for latency in latencies:
            if list(latency['latency']):
                if latency['class'] == 0:
                    pl.plot(latency['latency'][:1], 1, 'rx')
                elif latency['class'] == 1:
                    pl.plot(latency['latency'][:1], 2, 'gx')
                elif latency['class'] == 2:
                    pl.plot(latency['latency'][:1], 3, 'bx')
                elif latency['class'] == 3:
                    pl.plot(latency['latency'][:1], 4, 'bx')
                elif latency['class'] == 4:
                    pl.plot(latency['latency'][:1], 5, 'bx')
                elif latency['class'] == 5:
                    pl.plot(latency['latency'][:1], 6, 'bx')
                elif latency['class'] == 6:
                    pl.plot(latency['latency'][:1], 7, 'bx')
                elif latency['class'] == 7:
                    pl.plot(latency['latency'][:1], 8, 'bx')
                elif latency['class'] == 8:
                    pl.plot(latency['latency'][:1], 9, 'bx')
                elif latency['class'] == 9:
                    pl.plot(latency['latency'][:1], 10, 'bx')
        if show:
            pl.show()

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

        pl.plot(latency_paint['epoch'], latency_paint['latency'], 'b.')
        pl.xlabel('Epochs')
        pl.ylabel('Latency')
        pl.title(title)
        if show:
            pl.show()

    def plot_pattern(self, pattern, show=True):
        pl.clf()
        fig = pl.figure()
        fig.patch.set_facecolor('white')
        pl.ylim(0, 30)
        pl.xlim(0, len(pattern))
        pl.title('Temporal pattern')
        for neuron in range(len(pattern)):
            # if pattern[neuron]:
            pl.plot(neuron, pattern[neuron], 'b.')
        if show:
            pl.show()

    def plot_frequency_pattern(self, pattern, time, show=True):
        pl.ylim(0, len(pattern))
        pl.xlim(0, time)
        pl.title('Frequency pattern')
        for neuron in range(len(pattern)):
            pl.plot(pattern[neuron], [neuron] * len(pattern[neuron]), 'b.')
        if show:
            pl.show()

    def plot_image(self, image_spikes, image_size, title, show=True):
        # pl.xlim(0, image_size[0])
        # pl.ylim(0, image_size[1])

        scale = 100
        pl.title(title)

        spike_pos_x = 0
        spike_pos_y = image_size[1]
        new_image = []
        for neuron in image_spikes.keys():
            if image_spikes[neuron]:
                new_image.append(image_spikes[neuron][0])
            else:
                new_image.append(0)
        new_image = np.array(new_image).reshape(image_size)

        X = np.arange(0, image_size[0], 1)
        Y = np.arange(0, image_size[1], 1)
        X, Y = np.meshgrid(X, Y)

        pl.scatter(X, Y, new_image * scale)

        if show:
            pl.show()

    def plot_params(self, parameters_acc_pairs,
                    title='Parameters distribution',
                    normalize_colors=True, show=True):
        def get_polar(r, fi):
                x = r * cos(fi)
                y = r * sin(fi)
                return x, y

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

        pl.title(title)

        pl.ylim(-1.5, 1.5)
        pl.xlim(-1.5, 1.5)

        axes = get_axes(len(param_names))

        for (x, y), param in zip(axes, param_names):
            pl.plot(x, y, 'k-')
            pl.text(x[1], y[1], param)

        cmap = cm.get_cmap('viridis')
        for acc, parameters in sorted(zip(accs, all_params)):
            x, y = get_polars(parameters)
            if normalize_colors:
                color = cmap((acc - 0.8) / 0.2)
            else:
                color = cmap(acc)
            pl.plot(x, y, '-', color=color)
        if show:
            pl.show()

    def plot_pca(self, X, y, show=True):
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        Xpca = pca.fit_transform(X)

        pl.clf()
        fig = pl.figure()
        fig.patch.set_facecolor('white')
        
        # color_shape = ('r.', 'b.', 'g.', 'c.', 'k.', 'm.', 'y.', 'rx', 'bx', 'gx')
        classes_set = set(y)
        for cl in classes_set:
            class_mask = y == cl
            pl.plot(Xpca[class_mask].T[0], Xpca[class_mask].T[1], '.',
                    label='class ' + str(cl))
        pl.legend()
        if show:
            pl.show()
            
    def plot_latency_pca(self, X, y, max_val, show=True):
        from sklearn.decomposition import PCA

        latency_list = np.nan_to_num(X, nan=max_val)
        pca = PCA(n_components=2)
        Xpca = pca.fit_transform(latency_list)

        pl.clf()
        fig = pl.figure()
        fig.patch.set_facecolor('white')

        # color_shape = ('r.', 'b.', 'g.', 'c.', 'k.', 'm.', 'y.', 'rx', 'bx', 'gx')
        classes_set = set(y)
        for cl in classes_set:
            class_mask = y == cl
            pl.plot(Xpca[class_mask].T[0], Xpca[class_mask].T[1], '.',
                    label='class ' + str(cl))
        pl.legend()
        if show:
            pl.show()