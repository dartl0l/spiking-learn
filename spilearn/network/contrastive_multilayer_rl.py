# coding: utf-8

import nest
import numpy as np
from tqdm import tqdm

from .multilayer_rl import MultilayerRLNetwork


class ContrastiveMultilayerRLNetwork(MultilayerRLNetwork):
    """Multilayer RL network with explicit CSDP-style positive/negative samples.

    Extends :class:`MultilayerRLNetwork` (P2 v1) by feeding, for every training
    sample, an additional **negative** presentation built from a corrupted copy
    of the input. The hidden layer is driven by the graded goodness modulator
    ``m = label - p`` with ``label = 1`` on positives (push goodness up) and
    ``label = 0`` on negatives (push it down) — the contrastive / Forward-Forward
    objective, realized as reward-modulated STDP (CSDP Eq. 10). Negatives only
    shape the hidden representation: they do not update the output (classifier)
    layer and do not pollute the goodness EMA.

    Per original sample the input stream therefore holds four ``h_time`` windows:
    ``[positive(run), positive(learn), negative(run), negative(learn)]``.

    Parameters
    ----------
    negative_mode : str
        How to build a negative from a (latency-encoded) sample row:
        ``'shuffle'`` (permute features per row — same latency budget, broken
        structure) or ``'roll'`` (circular shift by half the width).
    negative_fn : callable or None
        Custom ``x -> x_neg`` (same shape). Overrides ``negative_mode``.
    """

    def __init__(
        self,
        settings,
        model,
        pool_size,
        learning_rate,
        learning_rate_hid,
        n_layer_hid,
        goodness_threshold_hid=None,
        negative_mode='shuffle',
        negative_fn=None,
        **kwargs,
    ):
        self.negative_mode = negative_mode
        self.negative_fn = negative_fn
        super().__init__(
            settings=settings,
            model=model,
            pool_size=pool_size,
            learning_rate=learning_rate,
            learning_rate_hid=learning_rate_hid,
            n_layer_hid=n_layer_hid,
            goodness_threshold_hid=goodness_threshold_hid,
            **kwargs,
        )

    def _make_negatives(self, x):
        x = np.asarray(x)
        if callable(self.negative_fn):
            return np.asarray(self.negative_fn(x))
        x_neg = x.copy()
        if self.negative_mode == 'shuffle':
            for row in x_neg:
                np.random.shuffle(row)
        elif self.negative_mode == 'roll':
            x_neg = np.roll(x_neg, x_neg.shape[1] // 2, axis=1)
        else:
            raise ValueError(f'unknown negative_mode={self.negative_mode!r}')
        return x_neg

    def run_negative(self):
        """Drive a negative input window; cache hidden energy without learning.

        Unlike ``run``, this does not update the goodness EMA threshold (the EMA
        should track positive/real data only) and does not predict a class.
        """
        self.update_learning_rate(0.0)
        self.sim()
        self.last_hidden_latency = self.get_raw_latency(self.spike_detector_hidden)
        self.last_hidden_energy = self.get_hidden_energy(self.last_hidden_latency)
        self.time += self.h_time

    def learn(self, action, correct_class):
        # positive presentation: hidden goodness target = 1
        hidden_learning_rate = (
            self.learning_rate_hid * self.hidden_goodness_modulator(1.0)
        )
        self.update_learning_rate(0)
        self.update_learning_rate_hid(hidden_learning_rate)
        self.update_learning_rate_separate_out(self.learning_rate, action)
        self.update_learning_rate_separate_out(self.learning_rate_scaled, correct_class)

        nest.Run(self.h_time)
        self.time += self.h_time

    def learn_negative(self):
        # negative presentation: hidden goodness target = 0, no output learning
        hidden_learning_rate = (
            self.learning_rate_hid * self.hidden_goodness_modulator(0.0)
        )
        self.update_learning_rate(0)
        self.update_learning_rate_hid(hidden_learning_rate)

        nest.Run(self.h_time)
        self.time += self.h_time

    def simulate(
        self,
        full_time,
        spike_dict,
        epochs=1,
        teacher_dicts=None,
        normalize_weights=False,
        y=None,
    ):
        progress_bar = tqdm(
            total=epochs * self.data_len * int(self.progress)
            + epochs * int(not self.progress),
            disable=not self.progress,
        )

        assert len(self.layers[1:]) == len(self.synapse_models), \
            "Number of layers and synapse models do not match."
        nest.Prepare()
        scale = 1
        running_accuracy = None
        for _ in range(epochs):
            self.spike_generator.set_input_spikes(spike_dict=spike_dict)

            accuracy_train = 0
            if y is not None:
                nest.Run(self.start_delta)
                self.time += self.start_delta
                for i in range(self.data_len):
                    target = y[i]

                    # --- positive presentation (run + learn, label = 1) ---
                    pred = self.run(self.n_layer_out)
                    if pred == target:
                        accuracy_train += 1
                        self.learning_rate = self.learning_rate_default * scale
                    else:
                        self.learning_rate = -1 * self.learning_rate_default * scale
                    self.learning_rate_scaled = self.learning_rate_default * scale
                    self.learn(pred, target)

                    # --- negative presentation (run + learn, label = 0) ---
                    self.run_negative()
                    self.learn_negative()

                    if normalize_weights and self.should_normalize(
                        i, self.normalize_step
                    ):
                        self.normalize(self.layers, self.synapse_models, self.w_targets)
                    progress_bar.update()
                accuracy_train /= self.data_len
                running_accuracy = self.ema_update(running_accuracy, accuracy_train)
                scale = self.lr_scale(running_accuracy)
            else:
                nest.Run(full_time)
                progress_bar.update()

            for spikes in spike_dict:
                spikes['spike_times'] += full_time

        nest.Cleanup()
        progress_bar.close()

    def train(self, x, y=None):
        self.init_network()
        self.create_layers()
        self.create_devices()

        self.connect_devices()
        self.connect_layers()

        if self.use_inhibition:
            self.connect_layers_inh()

        self.set_neuron_status(self.high_threshold_teacher)

        self.data_len = len(x)

        # interleave each sample with its negative, then repeat each twice for
        # the run/learn windows -> [pos, pos, neg, neg] per original sample
        x = np.asarray(x)
        x_neg = self._make_negatives(x)
        # preserve any trailing axes (TemporalPatternTransformer(reshape=True)
        # yields (n_samples, n_features, 1))
        interleaved = np.empty((x.shape[0] * 2,) + x.shape[1:], dtype=x.dtype)
        interleaved[0::2] = x
        interleaved[1::2] = x_neg
        x = np.repeat(interleaved, 2, axis=0)

        spike_dict = None
        full_time = None
        if self.spike_generator:
            self.spike_generator.create_devices()
            self.spike_generator.connect_devices(self.input_layer)

            spike_dict, full_time = self.spike_generator.create_spike_dict(
                dataset=x, tile=self.tile_train, delta=self.start_delta
            )

        teacher_dicts = None
        self.simulate(
            full_time,
            spike_dict,
            self.epochs,
            teacher_dicts,
            self.normalize_weights,
            y=y,
        )

        weights = self.save_weights(self.layers, self.synapse_models)
        output = {
            'spikes': nest.GetStatus(self.spike_detector_out, keys='events')[0][
                'times'
            ].tolist(),
            'senders': nest.GetStatus(self.spike_detector_out, keys='events')[0][
                'senders'
            ].tolist(),
        }
        devices = self.get_devices() if self.need_devices else None
        return weights, output, devices
