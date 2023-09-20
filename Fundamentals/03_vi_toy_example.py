# %%
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl

import sys
import os

sys.path.append(os.path.join('..', 'bll'))

import tools

from typing import List, Callable, Tuple, Optional

# %%

def get_output_noise_model(n_y: int) -> tf.keras.Model:
    """
    Returns a keras model that takes as input:
    - ``mu_y``: a tensor of shape (batch_size, n_y) with the mean of the output
    - ``rho_y``: a tensor of shape (batch_size, 1) which serves as a multiplicative 
    factor for the standard deviation of the output.
    The output is a distribution of shape (batch_size, n_y) with mean ``mu_y`` and
    standard deviation ``rho_y*sig_y``.
    """

    mu_y = tf.keras.Input(shape=(n_y,))
    rho_y = tf.keras.Input(shape=(1,))

    # sigma_y is created as a variable layer 
    sig_y = tfpl.VariableLayer(shape=(n_y,), initializer='zeros', activation=tf.math.exp)(rho_y)*rho_y

    cat_mu_y_and_sig_y = tf.keras.layers.Concatenate(axis=1)([mu_y, sig_y])

    dist = tfp.layers.DistributionLambda(
        lambda t: tfd.Independent(
        tfd.Normal(loc=t[..., :n_y], scale=t[..., n_y:]),
        reinterpreted_batch_ndims=1)
    )(cat_mu_y_and_sig_y)

    return keras.models.Model(inputs=[mu_y, rho_y], outputs=dist)


# %%

def trainable_mu_prior(kernel_size: int, bias_size: int, dtype=None) -> tf.keras.Model:
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype, initializer='normal'),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t,
                        scale=2*tf.ones(n)),
            reinterpreted_batch_ndims=1)),
    ])

def trainable_sig_prior(kernel_size: int, bias_size: int, dtype=None) -> tf.keras.Model:
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype, initializer='ones'),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=tf.zeros(n),
                        scale=tf.nn.softplus(t)),
            reinterpreted_batch_ndims=1)),
    ])

def prior(kernel_size: int, bias_size: int, dtype=None) -> tf.keras.Model:
    n = kernel_size + bias_size # num of params
    return tf.keras.Sequential([
       tfpl.DistributionLambda(
            lambda t: tfd.Independent(
                tfd.Normal(loc = tf.zeros(n), scale= 1*tf.ones(n)),
                reinterpreted_batch_ndims=1)
       )                     
  ])

# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior(kernel_size: int, bias_size: int, dtype=None) -> tf.keras.Model:
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype, initializer='normal'),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                scale=1e-5 + 0.003 * tf.math.exp(t[..., n:]-1)),
            reinterpreted_batch_ndims=1)),
    ])


_dist_type = Callable[[int, int, Optional[tf.dtypes.DType]], tf.keras.Model]


# %%

n_samples = 200
seed = 99

function_types = [1, 3]
sigma_noise = [5e-2, 2e-1]
n_channels = len(function_types)

train = tools.get_data(n_samples,[0,1], function_type=function_types, sigma=sigma_noise, dtype='float32', random_seed=seed)
test= tools.get_data(100, [-.4,1.4],   function_type=function_types, sigma=sigma_noise, dtype='float32', random_seed=seed)
true = tools.get_data(300, [-.4,1.4],  function_type=function_types, sigma=[0.,0.], dtype='float32')

train, val = tools.split(train, test_size=0.2)

# Create scaler from training data
scaler  = tools.Scaler(*train)

# Scale data (only required for testing purposes)
train_scaled = scaler.scale(*train)
val_scaled = scaler.scale(*val)
true_scaled = scaler.scale(*true)

def get_figure(n_channels=n_channels):
    # Plot data
    fig, ax = plt.subplots(n_channels, 1, figsize=(3.49, 3.0),dpi=150, sharex=True)

    if n_channels == 1:
        ax = [ax]

    for i in range(n_channels):
        ax[i].plot(true[0], true[1][:,i], label='true', color='k')
        ax[i].plot(train[0], train[1][:,i], 'x', color='k', label=f'train', alpha=0.5)
        ax[i].plot(test[0], test[1][:,i], '.', color='k', label=f'test', alpha=0.8)

    ax[0].set_ylabel('$y_1$')
    ax[1].set_ylabel('$y_2$')
    ax[-1].set_xlabel('$x$')

    return fig, ax

fig, ax = get_figure()
# %%
# Fix seeds
def get_bnn_model(m, full_bnn = True):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model_input = keras.Input(shape=(train[0].shape[1],))

    hidden_kwargs = {
        'units': 20,
        'activation': tf.nn.tanh,
        'make_prior_fn': prior,
        'make_posterior_fn': posterior,
        'kl_weight': 1/m,
    }

    output_kwargs = {
        'units': 2,
        'activation': 'linear',
        'make_prior_fn': trainable_sig_prior,
        'make_posterior_fn': posterior,
        'kl_weight': 1/m,
    }

    if full_bnn:
    # Hidden units
        architecture = [
            (tfpl.DenseVariational, hidden_kwargs),
            (tfpl.DenseVariational, hidden_kwargs),
            (tfpl.DenseVariational, hidden_kwargs),
        ]
    else:
        hidden_kwargs.pop('make_prior_fn')
        hidden_kwargs.pop('make_posterior_fn')
        hidden_kwargs.pop('kl_weight')
        architecture = [
            (tf.keras.layers.Dense, hidden_kwargs),
            (tf.keras.layers.Dense, hidden_kwargs),
            (tf.keras.layers.Dense, hidden_kwargs),
        ]

    architecture.append(
        (tfpl.DenseVariational, output_kwargs)
    )

    # Get layers and outputs:
    _, model_outputs = tools.DNN_from_architecture(model_input, architecture)
    output_model = keras.Model(inputs=model_input, outputs=model_outputs[-1])

    sigma_y_multiplier = tf.keras.Input(shape=(2,))
    output_with_noise = get_output_noise_model(2)([output_model.output, sigma_y_multiplier])

    output_with_noise_model = keras.Model(inputs=[model_input, sigma_y_multiplier], outputs=output_with_noise)


    return output_with_noise_model


negloglik = lambda y, p_y: -tf.reduce_mean(p_y.log_prob(y))

bnn_model = get_bnn_model(train[0].shape[0], full_bnn=True)

bnn_model.summary()


# %%
bnn_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.005),
    loss=negloglik,
    metrics=['mse'],
)
# %%

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='mse',
    patience=200,
    restore_best_weights=True,
)

hist = bnn_model.fit(
    x=[train_scaled[0], np.ones((train_scaled[0].shape[0], 1))],
    y=train_scaled[1],
    epochs=1000,
    verbose=1,
    callbacks=[early_stopping_callback],
)
# %%
unscale_std = lambda scaled: scaler.scaler_y.scale_*scaled
unscale = scaler.scaler_y.inverse_transform
samples = 10
Y_samp = [unscale(bnn_model([true_scaled[0], np.ones((true_scaled[0].shape[0],1))]).mean().numpy()) for _ in range(samples)]
Y_samp = np.stack(Y_samp, axis=2)
Y_std = unscale_std(bnn_model([true_scaled[0], np.ones((true_scaled[0].shape[0],1))]).stddev().numpy())
Y_std = np.repeat(Y_std[:,:, np.newaxis], samples, axis=2)

y_p3std = np.max(Y_samp + 3*Y_std, axis=2)
y_m3std = np.min(Y_samp - 3*Y_std, axis=2)

# %%
fig, ax = get_figure()
ax[0].plot(true[0], Y_samp[:,0,:], color='C0', alpha=0.5)
ax[1].plot(true[0], Y_samp[:,1,:], color='C0', alpha=0.5)
ax[0].fill_between(true[0].flatten(), y_m3std[:,0], y_p3std[:,0], color='C0', alpha=0.3)
ax[1].fill_between(true[0].flatten(), y_m3std[:,1], y_p3std[:,1], color='C0', alpha=0.3)
# %%
np.exp(bnn_model.layers[-1].trainable_variables[0].numpy())
# %%



# %%
