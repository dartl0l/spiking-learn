# coding: utf-8

import pylab as pl
import numpy as np
from math import cos, sin, pi
from matplotlib import animation, rc, cm
from mpl_toolkits.mplot3d import Axes3D


class Plotter:
    def __init__(self):
        pass

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
        pl.title('Weight distribution')
        for neuron in weights:
            pl.plot(list(range(len(weights[neuron]))),
                    weights[neuron], '.', label=str(neuron))
#         pl.legend()
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

    def plot_devices(self, devices, plot_last_detector=False):
        import nest.voltage_trace
        import nest.raster_plot

        nest.voltage_trace.from_device(devices['voltmeter'])
        pl.show()
        if plot_last_detector:
            nest.raster_plot.from_device(devices['spike_detector_hidden'], hist=False)
            pl.show()

        nest.raster_plot.from_device(devices['spike_detector_input'], hist=False)
        pl.show()

        nest.raster_plot.from_device(devices['spike_detector_out'], hist=False)
        pl.show()

    def plot_devices_limits(self, devices, start, end, plot_last_detector=False):
        import nest.voltage_trace
        import nest.raster_plot

        nest.voltage_trace.from_device(devices['voltmeter'])
        pl.xlim(start, end)
        pl.show()

        if plot_last_detector:
            nest.raster_plot.from_device(devices['spike_detector_hidden'], hist=False)
            pl.xlim(start, end)
            pl.show()

        nest.raster_plot.from_device(devices['spike_detector_input'], hist=False)
        pl.xlim(start, end)
        pl.show()

        nest.raster_plot.from_device(devices['spike_detector_out'], hist=False)
        pl.xlim(start, end)
        pl.show()

    def plot_latency(self, latency, classes, title, show=True):
        pl.clf()

        pl.title(title)
        colors = ['rx', 'gx', 'bx', 'cx', 'mx', 'yx', 'kx',
                  'ro', 'go', 'bo', 'co', 'mo', 'yo', 'ko']
#         shapes = ['x', 's', 'd']

        for one_latency, cl in zip(latency, classes):
            for i, neuron in enumerate(one_latency):
                pl.plot(one_latency[neuron][:1], i,
                        colors[cl], label=neuron)
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
        pl.ylim(0, 30)
        pl.xlim(0, len(pattern))
        pl.title('Temporal pattern')
        for neuron in range(len(pattern)):
            if pattern[neuron]:
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

    # def plot_pca(self, X, y, show=False):
    #     pca =
    #     X =
