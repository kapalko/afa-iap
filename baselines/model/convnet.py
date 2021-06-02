"""ConvNet that also takes vector observation: there are two input branches, one 
   for visual observation and the other one for vector observation"""
import numpy as np
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.framework import get_activation_fn, try_import_tf
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.policy.rnn_sequencing import add_time_dimension

tf1, tf, tfv = try_import_tf()


class CustomModel(RecurrentNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 vis_obs_shape, # visual observation shape
                 conv_filters, # conv layers' parameters: out_channel, kernel size, stride
                 conv_padding, # padding mode, 'same' or 'valid'
                 conv_activation, # conv layers' activation function
                 conv_final_activation, # activation function of the final conv layer
                 vec_obs_shape, # vector observation shape
                 fc_filters, # fc layers' parameters
                 fc_activation, # fc layers' activation function
                 fc_final_activation, # activation function of the final fc layer
                 merge_fc_filters, # fc layers' paramter in the "merge" network
                 merge_fc_activation, # activation function of the merge network
                 use_lstm, # whether to use lstm after merge network
                 lstm_cell_size, # lstm hidden dimension
                 max_seq_len):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.use_lstm = use_lstm

        # Network for visual observation
        vis_input = tf.keras.layers.Input(shape=vis_obs_shape,
                                          name="vis_observation")
        vis_feat = self._build_convnet(vis_input, conv_filters, conv_padding, 
                                       conv_activation, conv_final_activation, "visual")

        # Network for vector observation TODO: support multi-vector input
        vec_input = tf.keras.layers.Input(shape=vec_obs_shape,
                                          name="vec_observation")
        vec_feat = self._build_fcnet(vec_input, fc_filters, fc_activation, fc_final_activation, "non_visual")

        # Network for merging the two branch
        vis_feat_flat = tf.keras.layers.Flatten(data_format='channels_last')(vis_feat)
        cat_feats = [vis_feat_flat, vec_feat]
        cat_feats = tf.keras.layers.Concatenate(axis=-1)(cat_feats)

        merge_feat = self._build_fcnet(cat_feats, merge_fc_filters, merge_fc_activation, None, "merge")

        # Recurrent model
        if self.use_lstm:
            self.cell_size = lstm_cell_size
            state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="h")
            state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="c")
            seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

            max_seq_len = tf.shape(merge_feat)[0] // tf.shape(seq_in)[0]
            lstm_in = add_time_dimension(merge_feat, max_seq_len=max_seq_len, framework='tf')
            final_feat, state_h, state_c = tf.keras.layers.LSTM(
                self.cell_size,
                return_sequences=True,
                return_state=True,
                name="lstm")(
                    inputs=lstm_in,
                    mask=tf.sequence_mask(seq_in),
                    initial_state=[state_in_h, state_in_c])
        else:
            final_feat = merge_feat
        
        # Get policy and value function output
        act_output = tf.keras.layers.Dense(
            num_outputs,
            name='act_output',
            activation=None,
            kernel_initializer=normc_initializer(1.0)
        )(final_feat)

        if not self.model_config.get("vf_share_layers"):
            val_vis_feat = self._build_convnet(vis_input, conv_filters, conv_padding, 
                                               conv_activation, conv_final_activation, "value/visual")
            val_vec_feat = self._build_fcnet(vec_input, fc_filters, fc_activation, 
                                             fc_final_activation, "value/non_visual")

            val_vis_feat_flat = tf.keras.layers.Flatten(data_format='channels_last')(vis_feat)
            val_cat_feats = [val_vis_feat_flat, val_vec_feat]
            val_cat_feats = tf.keras.layers.Concatenate(axis=-1)(val_cat_feats)

            value_feat = self._build_fcnet(val_cat_feats, merge_fc_filters, merge_fc_activation, 
                                           None, "value/merge")

            if use_lstm:
                val_state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="value/h")
                val_state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="value/c")
                val_lstm_in = add_time_dimension(value_feat, max_seq_len=max_seq_len, framework='tf')
                value_feat, val_state_h, val_state_c = tf.keras.layers.LSTM(
                    self.cell_size,
                    return_sequences=True,
                    return_state=True,
                    name="value/lstm")(
                        inputs=val_lstm_in,
                        mask=tf.sequence_mask(seq_in),
                        initial_state=[val_state_in_h, val_state_in_c])
        else:
            value_feat = final_feat
        val_output = tf.keras.layers.Dense(
            1,
            name='val_output',
            activation=None,
            kernel_initializer=normc_initializer(1.0)
        )(value_feat)
            
        inputs = [vis_input, vec_input]
        outputs = [act_output, val_output]
        if self.use_lstm:
            if not self.model_config.get("vf_share_layers"):
                inputs += [state_in_h, state_in_c, val_state_in_h, val_state_in_c, seq_in]
                outputs += [state_h, state_c, val_state_h, val_state_c]
            else:
                inputs += [state_in_h, state_in_c, seq_in]
                outputs += [state_h, state_c]
        self.model = tf.keras.Model(inputs, outputs)
        self.register_variables(self.model.variables)
        
    def forward(self, input_dict, state, seq_lens):
        if self.use_lstm:
            outputs = self.model(input_dict["obs"] + state + [seq_lens])
            act_out, self._value_out = outputs[:2]
            act_out = tf.reshape(act_out, [-1, self.num_outputs])
            state = outputs[2:]
        else:
            act_out, self._value_out = self.model(input_dict["obs"])
        return act_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def get_initial_state(self):
        if not self.model_config.get("vf_share_layers"):
            init_states = [
                np.zeros(self.cell_size, np.float32),
                np.zeros(self.cell_size, np.float32),
                np.zeros(self.cell_size, np.float32),
                np.zeros(self.cell_size, np.float32),
            ]
        else:
            init_states = [
                np.zeros(self.cell_size, np.float32),
                np.zeros(self.cell_size, np.float32),
            ]
        return init_states

    def _build_convnet(self, inp, filters, padding, activation, final_activation, scope=None):
        activation = get_activation_fn(activation, framework="tf") if activation is not None else activation
        scope = scope if scope is None or scope[-1] == "/" else scope + "/"
        last_layer = inp
        for i, (out_c, kernel, stride) in enumerate(filters):
            activation_ = activation if i < len(filters) - 1 else None
            last_layer = tf.keras.layers.Conv2D(
                out_c,
                kernel,
                strides=(stride, stride),
                activation=activation_,
                padding=padding,
                data_format="channels_last",
                name="{}conv{}".format(scope, i))(last_layer)
        if final_activation is not None:
            final_activation = get_activation_fn(final_activation, framework="tf")
            last_layer = final_activation(last_layer)

        return last_layer

    def _build_fcnet(self, inp, filters, activation, final_activation, scope=None):
        activation = get_activation_fn(activation, framework="tf") if activation is not None else activation
        scope = scope if scope is None or scope[-1] == "/" else scope + "/"
        last_layer = inp
        for i, out_c in enumerate(filters):
            activation_ = activation if i < len(filters) - 1 else None
            last_layer = tf.keras.layers.Dense(
                out_c,
                activation=activation_,
                kernel_initializer=tf.initializers.variance_scaling(1.0),
                name="{}fc{}".format(scope, i))(last_layer)
        if final_activation is not None:
            final_activation = get_activation_fn(final_activation, framework="tf")
            last_layer = final_activation(last_layer)
        
        return last_layer
