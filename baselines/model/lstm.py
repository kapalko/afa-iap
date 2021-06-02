import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn, try_import_tf
from ray.rllib.models.tf.misc import normc_initializer

tf1, tf, tfv = try_import_tf()


class CustomModel(RecurrentNetwork):
    """An LSTM wrapper serving as an interface for ModelV2s that set use_lstm.
       Ref: https://github.com/ray-project/ray/blob/84617f6ff62561f857908f81e37f5d75a3215520/rllib/models/tf/recurrent_net.py
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, None,
                                          model_config, name)

        self.cell_size = model_config["lstm_cell_size"]
        if action_space.shape is not None:
            self.action_dim = int(np.product(action_space.shape))
        else:
            self.action_dim = int(len(action_space))

        self.num_outputs = num_outputs

        # Define input layers.
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")

        state_in_h = tf.keras.layers.Input(shape=(self.cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size, ), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            self.cell_size,
            return_sequences=True,
            return_state=True,
            name="lstm")(
                inputs=input_layer,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(lstm_out)
        if model_config["vf_share_layers"]:
            last_layer = lstm_out
        else:
            hiddens = model_config.get("fcnet_hiddens", [])
            activation = get_activation_fn(model_config.get("fcnet_activation"))
            last_layer = input_layer
            for i, size in enumerate(hiddens[:-1]):
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_layer)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values",
            kernel_initializer=normc_initializer(0.01))(last_layer)

        # Create the RNN model
        self._rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self._rnn_model.variables)
        self._rnn_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self._rnn_model([inputs, seq_lens] +
                                                           state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
