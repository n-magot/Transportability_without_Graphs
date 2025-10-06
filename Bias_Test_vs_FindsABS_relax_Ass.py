"""Supplementary Material, Synthetic data, relaxing the assumption of shared causal graph.
In this example we simulate data: De source, Do* target with binary outcome Y
- Causal graph in population  Π*:
                        X -> Y
                        Z -> Y
                        W -> Y
                    Confounders:
                        Z -> X
                        W -> X  (in Π this arrow does not exist)

P(Z)<>P*(Z) and P(W)<>P*(W)
"""

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


def Generate_DE_source(n, seed):
    np.random.seed(seed)
    coeffs = [np.random.uniform(-2.5, 2.5) for _ in range(3)]

    # Generate covariates: 'age', 'sex'
    e = np.random.normal(size=n)
    age = np.random.normal(1, 3, size=n)  # Normal distribution for age (mean=50, std=2)
    sex = np.random.binomial(1, 0.9, size=n)  # Binary feature for sex (50% chance for male or female)

    # x will include covariates: age, sex, comorbidity
    x_exp = np.column_stack((age, sex))

    # Treatment assignment (t): random binary treatment assignment (0 or 1)
    t_exp = np.random.choice([0, 1], size=n)

    log_odds = coeffs[0] * x_exp[:, 0] + coeffs[1] * sex + coeffs[2] * t_exp + e  # Log-odds for outcome

    # Apply logistic function (sigmoid) to convert log-odds to probabilities
    prob_y = 1 / (1 + np.exp(-log_odds))
    # Generate binary outcomes for y (0 or 1) based on the computed probabilities
    Y = np.random.binomial(1, prob_y)
    X_exp = np.column_stack((t_exp, x_exp))

    return X_exp, Y


def Generate_Do_target(n, seed):
    np.random.seed(seed)
    coeffs = [np.random.uniform(-2.5, 2.5) for _ in range(6)]

    # Generate covariates: 'age', 'sex', 'comorbidity'
    e = np.random.normal(size=n)
    age = np.random.normal(0, 5, size=n)  # Normal distribution for age (mean=50, std=10)
    sex = np.random.binomial(1, 0.3, size=n)  # Binary feature for sex (0 = male, 1 = female)

    # Stack the data into a DataFrame for each dataset
    # x will include covariates: age, sex, comorbidity
    x_obs = np.column_stack((age, sex))

    logit_t_obs = coeffs[3] + coeffs[4] * x_obs[:, 0] + coeffs[5] * sex  # Adjusted logit for treatment
    prob_t = 1 / (1 + np.exp(-logit_t_obs))  # Convert logit to probability
    t_obs = np.random.binomial(1, prob_t)  # Generate treatment assignment based on the probability

    # Outcome generation based on treatment and covariates (log-odds model)
    log_odds = coeffs[0] * x_obs[:, 0] + coeffs[1] * sex + coeffs[2] * t_obs + e  # Log-odds for outcome

    prob_y = 1 / (1 + np.exp(-log_odds))  # Convert log-odds to probability for binary outcome
    Y = np.random.binomial(1, prob_y)  # Generate binary outcome based on the probabilities
    X_obs = np.column_stack((t_obs, x_obs))

    return X_obs, Y


def binary_logistic_regression(data, labels):
    D = data.shape[1]
    alpha = numpyro.sample("alpha", dist.Cauchy(0, 10))
    beta = numpyro.sample("beta", dist.Cauchy(jnp.zeros(D), 2.5 * jnp.ones(D)))
    logits = alpha + jnp.dot(data, beta)
    return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)


def log_likelihood_calculation(alpha, beta, data, obs):
    logits = alpha + jnp.dot(data, beta)
    log_likelihood = dist.Bernoulli(logits=logits).log_prob(obs)
    return log_likelihood.sum()


def sample_prior(data, num_samples):
    prior_samples = {}
    D = data.shape[1]

    prior_samples1 = dist.Cauchy(jnp.zeros(D), 2.5 * jnp.ones(D)).sample(random.PRNGKey(0), (num_samples,))
    prior_samples2 = dist.Cauchy(jnp.zeros(1), 10 * jnp.ones(1)).sample(random.PRNGKey(0), (num_samples,))

    prior_samples["beta"] = prior_samples1
    prior_samples["alpha"] = prior_samples2

    return prior_samples


def sample_posterior(data, observed_data, num_samples):
    D = data.shape[1]

    kernel = NUTS(binary_logistic_regression)
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
        log_likelihoods = log_likelihoods.at[i].set(log_likelihood_calculation(samples["alpha"][i], samples["beta"][i],
                                                                               data, observed_data))
    # Estimate the log marginal likelihood using the log-sum-exp trick
    log_marginal_likelihood = jax.scipy.special.logsumexp(log_likelihoods) - jnp.log(num_samples)
    # print('marginal', log_marginal_likelihood)

    return log_marginal_likelihood


No = 5000
Ne_list = [50, 100, 300, 1000]

for Ne in Ne_list:
    columns = []
    df = pd.DataFrame(columns=columns)
    list_of_Hz_0 = []
    list_of_Hz_01 = []
    list_of_Hz_02 = []
    list_of_Hz_012 = []

    Log_loss_alg = []
    Log_loss_exp = []
    Log_loss_obs = []
    Log_loss_all = []

    seeds = np.arange(1, 21)  # 20 different seeds (from 1 to 20)

    # Run the functions with different seeds
    for i, seed in enumerate(seeds):

        data_exp, labels_exp = Generate_DE_source(Ne, seed)
        data_obs, labels_obs = Generate_Do_target(No, seed)

        print('how many 1 in observational', np.count_nonzero(labels_obs == 1))
        print('how many 1 in experimental', np.count_nonzero(labels_exp == 1))

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

            prior_samples = sample_prior(exp_sub_data, num_samples)

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
                CMB = reg_variables

            P_HZ[set] = math.exp(P_Hzc[set]) / (math.exp(P_Hzc[set]) + math.exp(P_Hz_not_c[set]))
            P_HZC[set] = 1 - P_HZ[set]
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

    print(df)
    print("PHz(X, )", list_of_Hz_0)
    print("PHz(X, Z1)", list_of_Hz_01)
    print("PHz(X, Z2)", list_of_Hz_02)
    print("PHz(X, Z1, Z2)", list_of_Hz_012)

    # Create a DataFrame
    AUC_results = pd.DataFrame({
        'PHz(T, )': list_of_Hz_0,
        'PHz(T, A)': list_of_Hz_01,
        'PHz(T, S)': list_of_Hz_02,
        'PHz(T, A, S)': list_of_Hz_012
    })

    # Save the DataFrame to a CSV file
    AUC_results.to_csv(f'AUC_random_results_N{Ne}.csv', index=False)
