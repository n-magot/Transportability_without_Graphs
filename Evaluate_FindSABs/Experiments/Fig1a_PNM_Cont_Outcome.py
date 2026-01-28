import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
from sklearn.metrics import log_loss
import pandas as pd
import math

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def random_continuous_coefficient():
    # Define the ranges
    ranges = [(0.1, 0.7)]
    # Choose a range randomly
    chosen_range = ranges[np.random.choice(len(ranges))]
    # Generate a random value within the chosen range
    return np.random.uniform(*chosen_range)

def Generate_De_source(sample_size, a_Z_Y, a_W_Y, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)
    # Parameters for linear combination
    beta0 = 0.5
    betaX = 1.0
    betaZ = a_Z_Y
    betaW = a_W_Y

    # Latent variables
    Z = np.random.normal(-2, 3, sample_size)  # target domain shift
    W = np.random.normal(-3, 3, sample_size)  # target domain shift
    e = np.random.normal(0, 0.05, sample_size)  # post-nonlinear noise for Y

    # Treatment
    X = np.random.binomial(1, 0.5, sample_size)

    # Invertible nonlinearities
    f1 = lambda x: x + 0.5 * x ** 3  # inner invertible
    f2 = lambda u: np.tanh(u)  # outer invertible

    # Canonical PNL: linear combination of all inputs + inner noise, then outer nonlinearity
    inner = beta0 + betaX * f1(X) + betaZ * Z + betaW * W + e
    Y = f2(inner)

    X_exp = np.column_stack((X, Z, W))
    return X_exp, Y


def Generate_Do_target(sample_size, a_Z_Y, a_W_Y, random_seed=None):
    """
    Generate data according to a canonical Post-Nonlinear (PNL) model:
        Y = f2( beta0 + betaX * f1(X) + betaZ * Z + betaW * W + e )
    f1 and f2 are invertible nonlinear functions.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Parameters for linear combination
    beta0 = 0.5
    betaX = 1.0
    betaZ = a_Z_Y
    betaW = a_W_Y

    # Latent variables
    Z = np.random.normal(0, 1, sample_size)
    W = np.random.normal(0, 1, sample_size)
    e = np.random.normal(0, 0.05, sample_size)  # inner noise

    # Treatment depends on Z
    p = 1 / (1 + np.exp(-Z))  # probability X=1
    X = np.random.binomial(1, p)

    # Invertible nonlinearities
    f1 = lambda x: x + 0.5 * x**3   # inner invertible
    f2 = lambda u: np.tanh(u)       # outer invertible

    # Canonical PNL: linear combination of all inputs + inner noise, then outer nonlinearity
    inner = beta0 + betaX * f1(X) + betaZ * Z + betaW * W + e
    Y = f2(inner)

    X_obs = np.column_stack((X, Z, W))

    return X_obs, Y


def Generate_De_target(sample_size, a_Z_Y, a_W_Y, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)

    # Parameters for linear combination
    beta0 = 0.5
    betaX = 1.0
    betaZ = a_Z_Y
    betaW = a_W_Y

    # Latent variables
    Z = np.random.normal(0, 1, sample_size)
    W = np.random.normal(0, 1, sample_size)
    e = np.random.normal(0, 0.05, sample_size)  # inner noise

    # Treatment depends on Z
    X = np.random.binomial(1, 0.5, sample_size)

    # Invertible nonlinearities
    f1 = lambda x: x + 0.5 * x ** 3  # inner invertible
    f2 = lambda u: np.tanh(u)  # outer invertible

    # Canonical PNL: linear combination of all inputs + inner noise, then outer nonlinearity
    inner = beta0 + betaX * f1(X) + betaZ * Z + betaW * W + e
    Y = f2(inner)

    X_exp = np.column_stack((X, Z, W))

    return X_exp, Y


def linear_regression(data, targets=None):
    D = data.shape[1]
    alpha = numpyro.sample("alpha", dist.Cauchy(0, 10))  # Intercept
    beta = numpyro.sample("beta", dist.Cauchy(jnp.zeros(D), 2.5 * jnp.ones(D)))  # Coefficients
    sigma = numpyro.sample("sigma", dist.HalfCauchy(5))  # Noise std dev

    mu = alpha + jnp.dot(data, beta)  # Linear predictor

    return numpyro.sample("obs", dist.Normal(mu, sigma), obs=targets)


def log_likelihood_linear(alpha, beta, sigma, data, obs):
    mu = alpha + jnp.dot(data, beta)
    log_likelihood = dist.Normal(mu, sigma).log_prob(obs)
    return log_likelihood.sum()


def sample_prior_linear(data, num_samples):
    prior_samples = {}
    D = data.shape[1]

    prior_samples["beta"] = dist.Cauchy(jnp.zeros(D), 10 * jnp.ones(D)).sample(random.PRNGKey(0), (num_samples,))
    prior_samples["alpha"] = dist.Cauchy(0, 2.5).sample(random.PRNGKey(0), (num_samples,))
    prior_samples["sigma"] = dist.HalfCauchy(5).sample(random.PRNGKey(0), (num_samples,))

    return prior_samples


def sample_posterior(data, observed_data, num_samples):
    D = data.shape[1]

    kernel = NUTS(linear_regression)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(jax.random.PRNGKey(42), data, observed_data)

    # Get the posterior samples
    posterior_samples = mcmc.get_samples()
    # import arviz as az
    # import matplotlib.pyplot as plt
    # data_plot = az.from_numpyro(mcmc)
    # az.plot_trace(data_plot, compact=True, figsize=(15, 25))
    # plt.show()

    return posterior_samples


def calculate_log_marginal(num_samples, samples, data, observed_data):
    log_likelihoods = jnp.zeros(num_samples)

    for i in range(num_samples):
        log_likelihoods = log_likelihoods.at[i].set(log_likelihood_linear(samples["alpha"][i], samples["beta"][i],
                                                                          samples["sigma"][i], data, observed_data))
    # Estimate the log marginal likelihood using the log-sum-exp trick
    log_marginal_likelihood = jax.scipy.special.logsumexp(log_likelihoods) - jnp.log(num_samples)
    # print('marginal', log_marginal_likelihood)

    return log_marginal_likelihood

# Example: list of 20 seeds
seeds = list(range(1, 2))  # or use random.sample for arbitrary seeds

No = 5000
Ne_list = [300]

for Ne in Ne_list:

    columns = []
    df = pd.DataFrame(columns=columns)
    list_of_Hz_0 = []
    list_of_Hz_01 = []
    list_of_Hz_02 = []
    list_of_Hz_012 = []

    for seed in seeds:
        print(f"\nRunning with seed: {seed}")
        np.random.seed(seed)

        """Let's assume that Z1 is a latent and that MB(Y)=(T,Z2)"""
        # Generate a new seed per run
        rs = np.random.seed(None)

        # Randomized parameters for outcome and treatment
        a_Z_Y = random_continuous_coefficient()
        a_W_Y = random_continuous_coefficient()

        # Generate source domain
        data_exp, labels_exp = Generate_De_source(
            sample_size=Ne,
            a_Z_Y=a_Z_Y,
            a_W_Y=a_W_Y,
            random_seed=rs  # do not reset seed inside function
        )

        # Generate observational domain
        data_obs, labels_obs = Generate_Do_target(
            sample_size=No,
            a_Z_Y=a_Z_Y,
            a_W_Y=a_W_Y,
            random_seed=rs
        )

        """Let's assume that Z1 is a latent and that MB(Y)=(T,Z2)"""
        P_Hzc = {}
        P_Hz_not_c = {}
        P_HZ = {}
        P_HZC = {}

        correct_IMB = 0
        num_samples = 1000

        """Searching in these subsets of MB"""
        subset_list = [(0,), (0, 1), (0, 2), (0, 1, 2)]


        for set in subset_list:
            reg_variables = set
            sub_data = data_obs[:, reg_variables]
            exp_sub_data = data_exp[:, reg_variables]

            posterior_samples = sample_posterior(sub_data, labels_obs, num_samples)

            prior_samples = sample_prior_linear(exp_sub_data, num_samples)

            marginal = calculate_log_marginal(num_samples, prior_samples, exp_sub_data, labels_exp)
            # print('Marginal {} from experimental sampling:'.format(reg_variables), marginal)
            """P(De|Do, Hzc_)"""
            P_Hz_not_c[reg_variables] = marginal

            marginal = calculate_log_marginal(num_samples, posterior_samples, exp_sub_data, labels_exp)
            # print('Marginal {} from observational sampling:'.format(reg_variables), marginal)
            """P(De|Do, Hzc)"""
            P_Hzc[reg_variables] = marginal

            if P_Hzc[reg_variables] > P_Hz_not_c[reg_variables]:
                print("CMB", reg_variables)
                print('P_HZ', P_Hzc[reg_variables])
                print('P_HZc', P_Hz_not_c[reg_variables])

                CMB = reg_variables

            #logg sum exp trick instead of:  P_HZ[set] = math.exp(P_Hzc[set]) / (math.exp(P_Hzc[set]) + math.exp(P_Hz_not_c[set]))
            diff = P_Hzc[set] - P_Hz_not_c[set]
            from scipy.special import expit

            P_HZ[set] = expit(diff)

            P_HZC[set] = 1 - P_HZ[set]

            # P_HZ[set] = math.exp(P_Hzc[set]) / (math.exp(P_Hzc[set]) + math.exp(P_Hz_not_c[set]))
            # P_HZC[set] = 1 - P_HZ[set]

            print('PHz for set', set, P_HZ[set])
            print('PHzc for set', set, P_HZC[set])

        # Generate dictionary with probabilities Hz and Hzc
        columns = []
        for tup in subset_list:
            columns.append(f'P_HZ{tup}')
            columns.append(f'P_HZc{tup}')

        # Find the set with the maximum P_Hz
        max_P_HZ = max(P_HZ, key=P_HZ.get)

        print('Set with the max P_Hz from FindsABS', max_P_HZ)

        # Create an empty DataFrame with the generated column names
        df = df.reindex(columns=columns)

        # Create the new_row list
        new_row = []
        for tpl in subset_list:
            new_row.append(P_HZ[tpl])  # Add value from P_HZ
            new_row.append(P_HZC[tpl])

        df.loc[len(df)] = new_row
        list_of_Hz_0.append(P_HZ[(0,)])
        list_of_Hz_01.append(P_HZ[(0, 1)])
        list_of_Hz_02.append(P_HZ[(0, 2)])
        list_of_Hz_012.append(P_HZ[(0, 1, 2)])

    print("PHz(X, )", list_of_Hz_0)
    print("PHz(X, Z1)", list_of_Hz_01)
    print("PHz(X, Z2)", list_of_Hz_02)
    print("PHz(X, Z1, Z2)", list_of_Hz_012)

    # Create a DataFrame
    AUC_results = pd.DataFrame({
        'PHz(X, )': list_of_Hz_0,
        'PHz(X, Z1)': list_of_Hz_01,
        'PHz(X, Z2)': list_of_Hz_02,
        'PHz(X, Z1, Z2)': list_of_Hz_012
    })


    # Save the DataFrame to a CSV file
    AUC_results.to_csv(f'2_AUC_PNM_Figure_1a_N{Ne}.csv', index=False)

