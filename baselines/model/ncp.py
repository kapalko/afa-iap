import numpy as np
import kerasncp as kncp

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn, try_import_tf
from ray.rllib.models.tf.misc import normc_initializer

tf1, tf, tfv = try_import_tf()


class CustomModel(RecurrentNetwork):
    """Neural circuit policy.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
                 inter_neurons, # Number of inter neurons
                 command_neurons, # Number of command neurons
                 sensory_fanout, # How many outgoing synapses has each sensory neuron
                 inter_fanout, # How many outgoing synapses has each inter neuron
                 recurrent_command_synapses, # Now many recurrent synapses are in the command neuron layer
                 motor_fanin # How many incoming syanpses has each motor neuron
                 ):
        super(CustomModel, self).__init__(obs_space, action_space, None,
                                          model_config, name)
        assert model_config["vf_share_layers"] == False

        if action_space.shape is not None:
            self.action_dim = int(np.product(action_space.shape))
        else:
            self.action_dim = int(len(action_space))

        self.cell_size = num_outputs + inter_neurons + command_neurons
        self.num_outputs = num_outputs

        # Define input layers.
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")

        state_in_h = tf.keras.layers.Input(shape=(self.cell_size, ), name="h")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LTC cell
        wiring = kncp.wirings.NCP(
            inter_neurons=inter_neurons,  
            command_neurons=command_neurons, 
            motor_neurons=self.num_outputs,  
            sensory_fanout=sensory_fanout,  
            inter_fanout=inter_fanout, 
            recurrent_command_synapses=recurrent_command_synapses, 
            motor_fanin=motor_fanin, 
        )
        rnn_cell = kncp.LTCCell(wiring)
        ltc_out, state_h = tf.keras.layers.RNN(
            rnn_cell, return_sequences=True, return_state=True, name="ltc")(
                inputs=input_layer,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h])
        logits = ltc_out # raw output of NCP is policy output

        # Independent network (FCs) for critic
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
            inputs=[input_layer, seq_in, state_in_h ],
            outputs=[logits, values, state_h])
        self.register_variables(self._rnn_model.variables)
        self._rnn_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h = self._rnn_model([inputs, seq_lens] +
                                                         state)
        return model_out, [h]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])