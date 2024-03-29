# %% [markdown]
# # Variational inference with Bayes by Backprop to train a Bayesian neural network
# Import the required packages.
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

export_figures = False
export_dir = '../Plots/MultivariateToyExample/'

# %% [markdown]
# # Generate and Display data for the investigated example

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

def bijection_std(x: tf.Tensor, a=1.0, b=0.0) -> tf.Tensor:
    """
    Returns a transformation of the input tensor x such that the output is positive.
    """
    return a * tf.math.exp(x+b)

# Define two parameterizations of the bijection function
bijection_std_output    = partial(bijection_std, a=1.0,   b=0.0)
bijection_std_posterior = partial(bijection_std, a=0.003, b=-2.0)

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
    sig_y = tfpl.VariableLayer(shape=(n_y,), initializer='zeros', activation=bijection_std_output)(rho_y)*rho_y

    cat_mu_y_and_sig_y = tf.keras.layers.Concatenate(axis=1)([mu_y, sig_y])

    dist = tfp.layers.DistributionLambda(
        lambda t: 
        tfd.Independent(
        tfd.Normal(loc=t[..., :n_y], scale=t[..., n_y:]),
        reinterpreted_batch_ndims=1)
    )(cat_mu_y_and_sig_y)

    return keras.models.Model(inputs=[mu_y, rho_y], outputs=dist)

# %% [markdown]
# # Variational inference with Bayes by Backprop
# To train a full BNN with Bayes by Backprop, 
# we need to specify a prior and a posterior distribution for each weight and bias in the network.
# We can then create the BNN with the ``tfp.layers.DenseVariational`` layer.
# %%

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
        # tfd.Independent(
                tfd.Normal(loc = tf.zeros(n), scale= .5*tf.ones(n)),
    #    reinterpreted_batch_ndims=1)
       )                  
  ])


# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior(kernel_size: int, bias_size: int, dtype=None) -> tf.keras.Model:
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype, initializer='normal'),
        tfp.layers.DistributionLambda(lambda t: 
            # tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                scale=bijection_std_posterior(t[..., n:])),
            # reinterpreted_batch_ndims=1)
            )
    ])

# %% [markdown]
# ## Create the BNN model  
# We create the same architecture as for the NN with BLL.

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

# %% [markdown]
# We can now get the model, check the architecture and compile it.
# For the compilation, we define the loss function as the negative log-likelihood of the data.

# %%

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
savename = '03_bnn_model_weights.h5'
savepath = os.path.join('.', 'results')

if not os.path.exists(os.path.join(savepath, savename)):
    hist = bnn_model.fit(
        x=[train_scaled[0], np.ones((train_scaled[0].shape[0], 1))],
        y=train_scaled[1],
        epochs=2000,
        verbose=0,
        batch_size=batch_size,
    )
    bnn_model.save_weights(os.path.join(savepath, savename))
else:
    bnn_model.load_weights(os.path.join(savepath, savename))

# %% [markdown]
# ## Plot the results
# For the plot we sample the posterior distribution
# of the weights and biases of the NN and predict the output for each sample.
# Additionally, we consider the output as a Gaussian mixture model and 
# compute its mean and standard deviation.

# %%
unscale_std = lambda scaled: scaler.scaler_y.scale_*scaled
unscale = scaler.scaler_y.inverse_transform
samples = 100

def sample_mean_and_std(model, x, samples=10):
    x_scaled =scaler.scale(X=x)[0]
    y_samp = [unscale(model([x_scaled, np.ones((x.shape[0],1))]).mean().numpy()) for _ in range(samples)]
    y_samp = np.stack(y_samp, axis=2)
    y_std = model([x, np.ones((x.shape[0],1))]).stddev().numpy()
    y_std = np.repeat(y_std[:,:, np.newaxis], samples, axis=2)
    return y_samp, y_std

tf.random.set_seed(99)

Y_samp, Y_std = sample_mean_and_std(bnn_model, true[0], samples=samples)

# Compute mean and standard deviation of the Gaussian mixture
Y_mix_mean = np.mean(Y_samp, axis=2)
Y_mix_std = np.mean(Y_std, axis=2) + np.mean(Y_samp**2, axis=2) -np.mean(Y_samp, axis=2)**2
y_mix_m_3std = Y_mix_mean - 3*Y_mix_std
y_mix_p_3std = Y_mix_mean + 3*Y_mix_std


fig, ax = get_figure()
ax[0].plot(true[0], Y_samp[:,0,:], color='C0', alpha=0.2)
ax[0].plot([],[], color='C0', alpha=0.2, label=r'$\bar{\vy}^{(i)}$')
ax[1].plot(true[0], Y_samp[:,1,:], color='C0', alpha=0.2)

ax[0].plot(true[0], Y_mix_mean[:,0], color='C1', alpha=1, label=r'$\bar{\vy}$')
ax[0].fill_between(true[0].flatten(), y_mix_m_3std[:,0], y_mix_p_3std[:,0], color='C1', alpha=0.3, label=r'$\pm 3\sigma$')

ax[1].plot(true[0], Y_mix_mean[:,1], color='C1', alpha=1)
ax[1].fill_between(true[0].flatten(), y_mix_m_3std[:,1], y_mix_p_3std[:,1], color='C1', alpha=0.3, label=r'$\pm 3\sigma$')

ax[0].legend()
ax[0].set_ylim([-2.2, 2.2])
ax[1].set_ylim([-2.2, 2.2])

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
def lpd_gmm(model, data, samples=100):
    Y_samp, Y_std = sample_mean_and_std(model, data[0], samples=samples)

    dY = Y_samp-data[1][:,:,np.newaxis]

    prob = np.exp(-.5*(dY / Y_std)**2)*1/(np.sqrt(2*np.pi)*Y_std)

    # Average over sample distribution
    prob_mean = np.mean(prob, axis=2)
    log_prob = np.log(prob_mean)

    return log_prob

tf.random.set_seed(99)
log_prob_train = lpd_gmm(bnn_model, train, samples=100)
log_prob_test = lpd_gmm(bnn_model, test, samples=100)

# %%

print(f'Log-probability of test data: {np.mean(log_prob_test):.3f}')
print(f'Log-probability of training data: {np.mean(log_prob_train):.3f}')


# %% [markdown]
# ## Investigating the posterior distribution of the weights
# We compare the prior and posterior distribution of the weights.
# This is conducted for an arbitrary layer of the network.

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


# %%
