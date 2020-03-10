import numpy as np
import tensorflow as tf

from robosuite.models.networks.layers import fc

def mlp(num_layers=2, hidden_dim=64, output_dim=64, activation=tf.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    hidden_dim: int                 size of fully-connected layers (default: 64)

    output_dim: int                 size of final fully-connected layer (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    return (features, recurrent features) here recurrent features is None
    """
    def network_fn(X):
        h = tf.layers.flatten(X)

        ## hidden layer
        for i in range(num_layers - 1):
            h = fc(h, 'mlp_fc{}'.format(i), nh=hidden_dim, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        h = fc(h, 'mlp_output', nh=output_dim, init_scale=np.sqrt(2))
        if layer_norm:
            h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
        h = activation(h)

        return (h, None)

    return network_fn