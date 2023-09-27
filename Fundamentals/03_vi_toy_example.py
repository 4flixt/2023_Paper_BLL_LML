# %%
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from functools import partial
import pandas as pd

from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl

import sys
import os

sys.path.append(os.path.join('..', 'bll'))
sys.path.append(os.path.join('..', 'Plots'))

import tools

from typing import List, Callable, Tuple, Optional
import config_mpl

export_figures = True
export_dir = '../Plots/MultivariateToyExample/'

# %% [markdown]
# # Toy example

# %%

n_samples = 200
seed = 99

function_types = [1, 3]
sigma_noise = [5e-2, 2e-1]
n_channels = len(function_types)

train = tools.get_data(n_samples,[0,1], function_type=function_types, sigma=sigma_noise, dtype='float32', random_seed=seed)
test= tools.get_data(100, [-.5,1.5],   function_type=function_types, sigma=sigma_noise, dtype='float32', random_seed=seed)
true = tools.get_data(300, [-.5,1.5],  function_type=function_types, sigma=[0.,0.], dtype='float32')

train, val = tools.split(train, test_size=0.2)

# Create scaler from training data
scaler  = tools.Scaler(*train)

# Scale data (only required for testing purposes)
train_scaled = scaler.scale(*train)
test_scaled = scaler.scale(*test)
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

get_figure()

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
        lambda t: 
        tfd.Independent(
        tfd.Normal(loc=t[..., :n_y], scale=t[..., n_y:]),
        reinterpreted_batch_ndims=1)
    )(cat_mu_y_and_sig_y)

    return keras.models.Model(inputs=[mu_y, rho_y], outputs=dist)


# %%

def bijection_std(x: tf.Tensor, a=1.0, b=0.0) -> tf.Tensor:
    """
    Returns a transformation of the input tensor x such that the output is positive.
    """
    return a * tf.math.exp(x+b)

# Define two parameterizations of the bijection function
bijection_std_output    = partial(bijection_std, a=1.0,   b=0.0)
bijection_std_posterior = partial(bijection_std, a=0.003, b=-2.0)


def trainable_sig_prior(kernel_size: int, bias_size: int, dtype=None) -> tf.keras.Model:
    """
    Returns a keras model to represent a prior distribution 
    for the model weights (kernel) and biases. 
    The distribution is a Normal distribution with trainable variance.
    
    - The trainable parameter theta is initialized to zero and is transformed to be positive with 
    the function exp(theta). In this way, the prior is initially a normal distribution with mean zero and variance one.
    """
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype, initializer='zeros'),
        tfp.layers.DistributionLambda(lambda t: 
            # tfd.Independent(
            tfd.Normal(loc=tf.zeros(n),
                        scale=bijection_std_output(t)),
            # reinterpreted_batch_ndims=1)
            ),
    ])


def prior(kernel_size: int, bias_size: int, dtype=None) -> tf.keras.Model:
    """
    Returns a keras model to represent a prior distribution 
    for the model weights (kernel) and biases. 
    The distribution is a Normal distribution with fixed mean and variance.
    """
    n = kernel_size + bias_size # num of params
    return tf.keras.Sequential([
       tfpl.DistributionLambda(
            lambda t: 
        tfd.Independent(
                tfd.Normal(loc = tf.zeros(n), scale= .5*tf.ones(n)),
       reinterpreted_batch_ndims=1)
       )                  
  ])


# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior(kernel_size: int, bias_size: int, dtype=None) -> tf.keras.Model:
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype, initializer='normal'),
        tfp.layers.DistributionLambda(lambda t: 
            tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                scale=bijection_std_posterior(t[..., n:])),
            reinterpreted_batch_ndims=1)
            )
    ])

# %%
# Fix seeds
def get_bnn_model(m, full_bnn = True):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    n_x = train[0].shape[1]
    n_y = train[1].shape[1]

    model_input = keras.Input(shape=(n_x,))

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
        'make_prior_fn': prior,
        'make_posterior_fn': posterior,
        'kl_weight': 1/m,
    }

    if full_bnn:
    # Hidden units
        architecture = [
            (tfpl.DenseVariational, hidden_kwargs),
            (keras.layers.BatchNormalization, {}),
            (tfpl.DenseVariational, hidden_kwargs),
            (keras.layers.BatchNormalization, {}),
            (tfpl.DenseVariational, hidden_kwargs),
            (keras.layers.BatchNormalization, {}),
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

    # The sigma_y multiplier is an untrainable input that 
    # scaled the standard deviation for each training point. 
    # Typically, 1 is used for all points.
    sigma_y_multiplier = tf.keras.Input(shape=(n_y,))
    output_with_noise = get_output_noise_model(n_y)([output_model.output, sigma_y_multiplier])

    output_with_noise_model = keras.Model(inputs=[model_input, sigma_y_multiplier], outputs=output_with_noise)


    return output_with_noise_model



n_batches = 1
batch_size = train[0].shape[0]//n_batches


negloglik = lambda y, p_y: -n_batches*tf.reduce_mean(p_y.log_prob(y))

bnn_model = get_bnn_model(batch_size, full_bnn=True)

bnn_model.summary()

# %%
bnn_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    loss=negloglik,
    metrics=['mse'],
)
# %%

savename = '02_bnn_model_weights.h5'
savepath = os.path.join('.', 'results')
if False:
    hist = bnn_model.fit(
        x=[train_scaled[0], np.ones((train_scaled[0].shape[0], 1))],
        y=train_scaled[1],
        epochs=2000,
        verbose=0,
        batch_size=batch_size,
        # callbacks=[early_stopping_callback],
    )
    bnn_model.save_weights(os.path.join(savepath, savename))
else:
    bnn_model.load_weights(os.path.join(savepath, savename))


# %%
unscale_std = lambda scaled: scaler.scaler_y.scale_*scaled
unscale = scaler.scaler_y.inverse_transform
samples = 50

def sample_mean_and_std(model, x, samples=10):
    y_samp = [model([x, np.ones((x.shape[0],1))]).mean().numpy() for _ in range(samples)]
    y_samp = np.stack(y_samp, axis=2)
    y_std = model([x, np.ones((x.shape[0],1))]).stddev().numpy()
    y_std = np.repeat(y_std[:,:, np.newaxis], samples, axis=2)
    return y_samp, y_std


Y_samp, Y_std = sample_mean_and_std(bnn_model, true_scaled[0], samples=samples)

y_p3std = np.max(Y_samp + 3*Y_std, axis=2)
y_m3std = np.min(Y_samp - 3*Y_std, axis=2)
# %%
fig, ax = get_figure()
ax[0].plot(true[0], Y_samp[:,0,:], color='C0', alpha=0.2)
ax[0].plot([],[], color='C0', alpha=0.2, label=r'$\bar{\vy}^{(i)}$')
ax[1].plot(true[0], Y_samp[:,1,:], color='C0', alpha=0.2)
ax[0].fill_between(true[0].flatten(), y_m3std[:,0], y_p3std[:,0], color='C0', alpha=0.3, label=r'$\pm 3\sigma$')
ax[1].fill_between(true[0].flatten(), y_m3std[:,1], y_p3std[:,1], color='C0', alpha=0.3)
# for i in range(samples):
    # ax[0].fill_between(true[0].flatten(), Y_samp[:,0,i]-3*Y_std[:,0,i],Y_samp[:,0,i]-3*Y_std[:,0,i], color='C0', alpha=0.1, edgecolor=None)
    # ax[1].fill_between(true[0].flatten(), Y_samp[:,1,i]-3*Y_std[:,1,i],Y_samp[:,1,i]-3*Y_std[:,1,i], color='C0', alpha=0.1, edgecolor=None)

ax[0].legend()

fig.align_ylabels()
legend = ax[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.5), fancybox=True, framealpha=1)
fig.tight_layout(pad=0.2)


if export_figures:
    name = 'bnn_vi_toy_example'
    fig.savefig(f'{export_dir}{name}.pdf')
    fig.savefig(f'{export_dir}{name}.pgf')


# %% [markdown]
# ## Check estimated noise variances

# %%
est_sig_y = unscale_std(bijection_std_output(bnn_model.layers[-1].trainable_variables[0]).numpy())

print(f'Estimated noise std: {np.round(est_sig_y, 3)}')
print(f'True noise std:      {np.round(sigma_noise, 3)}')
# %% [markdown]
"""
## Compute the log-predictive density of the test set
"""

# %%

samples = 100
Y_samp, Y_std = sample_mean_and_std(bnn_model, test_scaled[0], samples=samples)

dY = Y_samp-test[1][:,:,np.newaxis]

prob = np.exp(-.5*(dY / Y_std)**2)*1/(np.sqrt(2*np.pi)*Y_std)

# Average over sample distribution
prob_mean = np.mean(prob, axis=2)
log_prob = np.log(prob_mean)

print(f'Log-probability of test data: {np.mean(log_prob):.3f}')

# %% 
samples = 100
Y_samp, Y_std = sample_mean_and_std(bnn_model, train_scaled[0], samples=samples)

dY = Y_samp-train[1][:,:,np.newaxis]

prob = np.exp(-.5*(dY / Y_std)**2)*1/(np.sqrt(2*np.pi)*Y_std)

# Average over sample distribution
prob_mean = np.mean(prob, axis=2)
log_prob = np.log(prob_mean)

print(f'Log-probability of training data: {np.mean(log_prob):.3f}')


# %% [markdown]
# ## Understanding the prior and posterior distributions

# %%

w0 = bnn_model.layers[3].weights[0]
n = w0.shape[0]//2
mu_w0 = w0[:n]
rho_w0 = w0[n:]
sig_w0 = bijection_std_posterior(rho_w0)


fig, ax = plt.subplots()

df = pd.DataFrame({'mean': np.abs(mu_w0.numpy()), 'stddev': sig_w0.numpy()})
df.plot(x='stddev', y='mean', kind='scatter', ax=ax, label='posterior', color='C0')
ax.scatter(0.5, 0, color='C1', label='prior', s=100, marker='x')
ax.legend()
ax.set_title('Posterior vs prior distribution of weights')

# %% [markdown]
# Check if log-prob is computing what I expect ... 

# %%
tf.random.set_seed(00)
y_pred =bnn_model([train_scaled[0], np.ones((train_scaled[0].shape[0],1))])
tf.reduce_mean(y_pred.log_prob(train_scaled[1]))
# %%
lp = -.5*((y_pred.mean()-train_scaled[1])/y_pred.stddev())**2-.5*np.log(2*np.pi*y_pred.stddev()**2)
np.mean(np.sum(lp,axis=1))
# %%

