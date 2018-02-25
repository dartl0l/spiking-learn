import nest.raster_plot
import nest.voltage_trace
import pylab as pl
import numpy as np
from matplotlib import animation, rc


class Plotter:

    def __init__(self):
        pass

    def plot_field(self, sigma2, max_x, n_fields):

        def get_gaussian(x, sigma2, mu):
            return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma2))

        h_mu = max_x / (n_fields - 1)
        mu = 0
        for _ in xrange(n_fields):
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
            pl.plot(range(len(weights[neuron])),
                    weights[neuron], '.', label=str(neuron))
        pl.legend()
        if show:
            pl.show()
        else:
            return pl.plot()

    def plot_animated_weights(self, weights_history, h, save, show):
        def plot_weights_for_anim(self, weights):
            neurons = weights.keys()
            pl.title('Weight distribution')

            plot = pl.plot(range(len(weights[neurons[0]])), weights[neurons[0]], 'r.',
                           range(len(weights[neurons[1]])), weights[neurons[1]], 'b.')

            return plot

        all_plot = []
        fig = pl.figure()

        for i in xrange(0, len(weights_history), h):
            weights = weights_history[i]
            all_plot.append(plot_weights_for_anim(weights['layer_0']))

        weights_anim = animation.ArtistAnimation(fig, all_plot, blit=True)

        if save is True:
            weights_anim.save('weights.mp4')
        #     if show is True:
        #         HTML(weights_anim.to_html5_video())
        return weights_anim

    def plot_devices(self, devices):
        nest.voltage_trace.from_device(devices['voltmeter'])
        pl.show()

        nest.raster_plot.from_device(devices['spike_detector_2'], hist=False)
        pl.show()

        nest.raster_plot.from_device(devices['spike_detector_1'], hist=False)
        pl.show()

    def plot_devices_start(self, devices, settings):
        nest.voltage_trace.from_device(devices['voltmeter'])
        pl.xlim(settings['start_delta'], settings['start_delta'] + settings['h_time'])
        pl.show()

        nest.raster_plot.from_device(devices['spike_detector_2'], hist=False)
        pl.xlim(settings['start_delta'], settings['start_delta'] + settings['h_time'])
        pl.show()

        nest.raster_plot.from_device(devices['spike_detector_1'], hist=False)
        pl.xlim(settings['start_delta'], settings['start_delta'] + settings['h_time'])
        pl.show()

    def plot_devices_end(self, devices, settings):
        nest.voltage_trace.from_device(devices['voltmeter'])
        pl.xlim(settings['full_time'] - settings['h_time'], settings['full_time'])
        pl.show()

        nest.raster_plot.from_device(devices['spike_detector_2'], hist=False)
        pl.xlim(settings['full_time'] - settings['h_time'], settings['full_time'])
        pl.show()

        nest.raster_plot.from_device(devices['spike_detector_1'], hist=False)
        pl.xlim(settings['full_time'] - settings['h_time'], settings['full_time'])
        pl.show()

    def plot_latencies(self, latencies):
        pl.title('Output latencies')
        for latency in latencies:
            if list(latency['latency']):
                if latency['class'] == 0:
                    pl.plot(latency['latency'][:1], 1, 'rx')
                elif latency['class'] == 1:
                    pl.plot(latency['latency'][:1], 2, 'gx')
                elif latency['class'] == 2:
                    pl.plot(latency['latency'][:1], 3, 'bx')
        pl.show()

    def plot_train_latency(self, latency_train):
        latency_paint = {'latency': [],
                         'epoch': []}

        epoch = 1

        for latency in latency_train:
            if latency['latency']:
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
        pl.show()

    def plot_pattern(self, pattern):
        pl.ylim(0, 30)
        pl.xlim(0, len(pattern))
        pl.title('Temporal pattern')
        for neuron in pattern.keys():
            if pattern[neuron]:
                pl.plot(neuron, pattern[neuron], 'b.')
        pl.show()
