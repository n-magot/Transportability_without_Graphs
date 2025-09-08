"""Experimental section, Scenario 1, UAI2025
In this example we simulate data: Do, De, Do* with binary outcome Y
- Do in the source population, P:
                        X -> Y
                        Z -> Y
                        W -> Y
                        Z1,...,Z7 -> Y
                    Confounders:
                        Z -> X
                        W -> X  (in the target population, P*, this arrow does not exist)

P(Z)<>P*(Z) and P(W)<>P*(W)
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from fontTools.misc.bezierTools import epsilon
from numpyro.infer import MCMC, NUTS
from jax import random
from sklearn.metrics import log_loss
import pandas as pd
import math
from itertools import combinations


# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def Generate_DE_source(sample_size, intercept_Y, b_T_Y, b_Z2_Y, b_Z1_Y, Z1_mean_P, Z1_std_P, Z2_mean_P, Z2_std_P,
                       random_seed=None):
    e = np.random.normal(size=sample_size)
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate independent binary variables T and Z2
    Z1 = np.random.normal(Z1_mean_P, Z1_std_P, sample_size)
    Z2 = np.random.normal(Z2_mean_P, Z2_std_P, sample_size)
    T = np.random.binomial(1, 0.5, sample_size)  # T ~ Bernoulli(0.5)
    Z3 = np.random.normal(3, 6, sample_size)
    Z4 = np.random.normal(0, 3, sample_size)
    Z5 = np.random.normal(-1, 4, sample_size)
    Z6 = np.random.normal(-5, 5, sample_size)
    Z7 = np.random.binomial(1, 0.5, sample_size)  # T ~ Bernoulli(0.5)
    Z8 = np.random.binomial(1, 0.8, sample_size)  # T ~ Bernoulli(0.5)
    Z9 = np.random.binomial(1, 0.3, sample_size)  # T ~ Bernoulli(0.5)

    b_Z3_Y, b_Z4_Y, b_Z5_Y, b_Z6_Y, b_Z7_Y, b_Z8_Y, b_Z9_Y = 0.06, 0.09, 0.1, 0.2, 1.5, -1, -0.9
    # Generate Y based on the logistic model
    logit_Y = (intercept_Y + b_T_Y * T + b_Z1_Y * Z1 + b_Z2_Y * Z2 + b_Z3_Y * Z3 + b_Z4_Y * Z4 + b_Z5_Y * Z5 +
               b_Z6_Y * Z6 + b_Z7_Y * Z7 + b_Z8_Y * Z8+ b_Z9_Y * Z9+ e)
    pr_Y = np.clip(1 / (1 + np.exp(-logit_Y)), 1e-6, 1 - 1e-6)
    Y = np.random.binomial(1, pr_Y, sample_size)

    X_exp = np.column_stack((T, Z3, Z4, Z5, Z6, Z7, Z8, Z9))

    return X_exp, Y


def Generate_Do_target(sample_size, intercept_Y, b_T_Y, b_Z2_Y, b_Z1_Y, intercept_T, b_Z2_T, Z1_mean_PT, Z1_std_PT,
                       Z2_mean_PT, Z2_std_PT, random_seed=None):
    e = np.random.normal(size=sample_size)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate Z2 (strong confounder)
    Z1 = np.random.normal(Z1_mean_PT, Z1_std_PT, sample_size)
    Z2 = np.random.normal(Z2_mean_PT, Z2_std_PT, sample_size)

    # Generate T based on Z2
    logit_T = intercept_T + b_Z2_T * Z2
    prob_T = np.clip(1 / (1 + np.exp(-logit_T)), 1e-6, 1 - 1e-6)
    # print(prob_T)
    T = np.random.binomial(1, prob_T)

    Z3 = np.random.normal(3, 6, sample_size)
    Z4 = np.random.normal(0, 3, sample_size)
    Z5 = np.random.normal(-1, 4, sample_size)
    Z6 = np.random.normal(-5, 5, sample_size)
    Z7 = np.random.binomial(1, 0.5, sample_size)  # T ~ Bernoulli(0.5)
    Z8 = np.random.binomial(1, 0.8, sample_size)  # T ~ Bernoulli(0.5)
    Z9 = np.random.binomial(1, 0.3, sample_size)  # T ~ Bernoulli(0.5)

    # Generate Y based on the logistic model
    b_Z3_Y, b_Z4_Y, b_Z5_Y, b_Z6_Y, b_Z7_Y, b_Z8_Y, b_Z9_Y = 0.06, 0.09, 0.1, 0.2, 1.5, -1, -0.9
    # Generate Y based on the logistic model
    logit_Y = (intercept_Y + b_T_Y * T + b_Z1_Y * Z1 + b_Z2_Y * Z2 + b_Z3_Y * Z3 + b_Z4_Y * Z4 + b_Z5_Y * Z5 +
               b_Z6_Y * Z6 + b_Z7_Y * Z7 + b_Z8_Y * Z8 + b_Z9_Y * Z9 + e)
    pr_Y = np.clip(1 / (1 + np.exp(-logit_Y)), 1e-6, 1 - 1e-6)
    Y = np.random.binomial(1, pr_Y, sample_size)

    X_obs = np.column_stack((T, Z3, Z4, Z5, Z6, Z7, Z8, Z9))

    return X_obs, Y


def Generate_De_target(sample_size, intercept_Y, b_T_Y, b_Z2_Y, b_Z1_Y, Z1_mean_PT, Z1_std_PT, Z2_mean_PT, Z2_std_PT,
                       random_seed=None):
    e = np.random.normal(size=sample_size)

    if random_seed is not None:
        np.random.seed(random_seed)

    Z1 = np.random.normal(Z1_mean_PT, Z1_std_PT, sample_size)
    Z2 = np.random.normal(Z2_mean_PT, Z2_std_PT, sample_size)

    T = np.random.binomial(1, 0.5, sample_size)  # T ~ Bernoulli(0.5)

    # Generate Y based on the logistic model
    Z3 = np.random.normal(3, 6, sample_size)
    Z4 = np.random.normal(0, 3, sample_size)
    Z5 = np.random.normal(-1, 4, sample_size)
    Z6 = np.random.normal(-5, 5, sample_size)
    Z7 = np.random.binomial(1, 0.5, sample_size)  # T ~ Bernoulli(0.5)
    Z8 = np.random.binomial(1, 0.8, sample_size)  # T ~ Bernoulli(0.5)
    Z9 = np.random.binomial(1, 0.3, sample_size)  # T ~ Bernoulli(0.5)

    # Generate Y based on the logistic model
    b_Z3_Y, b_Z4_Y, b_Z5_Y, b_Z6_Y, b_Z7_Y, b_Z8_Y, b_Z9_Y = 0.06, 0.09, 0.1, 0.2, 1.5, -1, -0.9
    # Generate Y based on the logistic model
    logit_Y = (intercept_Y + b_T_Y * T + b_Z1_Y * Z1 + b_Z2_Y * Z2 + b_Z3_Y * Z3 + b_Z4_Y * Z4 + b_Z5_Y * Z5 +
               b_Z6_Y * Z6 + b_Z7_Y * Z7 + b_Z8_Y * Z8 + b_Z9_Y * Z9 + e)
    pr_Y = np.clip(1 / (1 + np.exp(-logit_Y)), 1e-6, 1 - 1e-6)
    Y = np.random.binomial(1, pr_Y, sample_size)

    X_exp = np.column_stack((T, Z3, Z4, Z5, Z6, Z7, Z8, Z9))

    return X_exp, Y


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


def random_binary_coefficient():
    # Define the ranges
    ranges = [(-2.5, -0.5), (0.5, 2.5)]
    # Choose a range randomly
    chosen_range = ranges[np.random.choice(len(ranges))]
    # Generate a random value within the chosen range
    return np.random.uniform(*chosen_range)


def random_continuous_coefficient():
    # Define the ranges
    ranges = [(-1, -0.2), (0.2, 1)]
    # Choose a range randomly
    chosen_range = ranges[np.random.choice(len(ranges))]
    # Generate a random value within the chosen range
    return np.random.uniform(*chosen_range)


def generate_random_params():
    # Randomly sample mean1 and std1 from (0, 10)
    mean1 = np.random.uniform(-5, 5)
    std1 = np.random.uniform(0, 5)

    # Randomly sample mean2, ensuring it's different from mean1
    mean2 = mean1
    while mean2 == mean1:
        mean2 = np.random.uniform(-5, 5)

    # Randomly sample std2, ensuring it's different from std1
    std2 = std1
    while std2 == std1:
        std2 = np.random.uniform(0, 5)

    return mean1, std1, mean2, std2

def score_func(reg_variables):

    sub_data = data_obs[:, reg_variables]
    exp_sub_data = data_exp[:, reg_variables]

    posterior_samples = sample_posterior(sub_data, labels_obs, num_samples)

    prior_samples = sample_prior(exp_sub_data, num_samples)

    marginal_Hz_not_c = calculate_log_marginal(num_samples, prior_samples, exp_sub_data, labels_exp)
    # print('Marginal {} from experimental sampling:'.format(reg_variables), marginal
    """P(De|Do, Hzc_)"""
    P_Hz_not_c[reg_variables] = marginal_Hz_not_c

    marginal_Hz_c = calculate_log_marginal(num_samples, posterior_samples, exp_sub_data, labels_exp)

    # print('Marginal {} from observational sampling:'.format(reg_variables), marginal)
    """P(De|Do, Hzc)"""
    P_Hzc[reg_variables] = marginal_Hz_c

    if P_Hzc[reg_variables] > P_Hz_not_c[reg_variables]:
        print("CMB", reg_variables)

    x = P_Hzc[reg_variables]
    y = P_Hz_not_c[reg_variables]

    # Safe sigmoid computation
    P_HZ[reg_variables] = 0.5 * (1 + math.tanh((x - y) / 2))

    P_HZC[reg_variables] = 1 - P_HZ[reg_variables]
    print('PHz for set', reg_variables, P_HZ[reg_variables])
    print('PHzc for set', reg_variables, P_HZC[reg_variables])

    return marginal_Hz_c, P_HZ, P_HZC


No = 5000
Ne_list = [2000]
N_test = 1000
for Ne in Ne_list:
    n_runs = 20  # Number of different datasets that you want to test the algorithm

    list_of_Hz_0 = []
    list_of_Hz_01 = []
    list_of_Hz_02 = []
    list_of_Hz_012 = []

    Log_loss_alg = []
    Log_loss_exp = []
    Log_loss_obs = []
    Log_loss_all = []

    for k in range(n_runs):

        P_HAT_alg = {}
        P_HAT_exp = {}
        P_HAT_obs = {}
        P_HAT_all = {}

        # Experimental data coefficients
        intercept_Y = random_binary_coefficient()  # Intercept for Y
        b_T_Y = random_binary_coefficient()  # Effect of T on Y
        b_Z2_Y = random_continuous_coefficient()  # Effect of Z2 on Y
        b_Z1_Y = random_continuous_coefficient()  # Effect of Z1 on Y
        Z1_mean_P, Z1_std_P, Z1_mean_PT, Z1_std_PT = generate_random_params()
        Z2_mean_P, Z2_std_P, Z2_mean_PT, Z2_std_PT = generate_random_params()
        rs = np.random.seed(None)

        # Generate experimental data
        data_exp, labels_exp = Generate_DE_source(
            sample_size=Ne,
            intercept_Y=intercept_Y,
            b_T_Y=b_T_Y,
            b_Z1_Y=b_Z1_Y,
            b_Z2_Y=b_Z2_Y,
            Z1_mean_P=Z1_mean_P,
            Z1_std_P=Z1_std_P,
            Z2_mean_P=Z2_mean_P,
            Z2_std_P=Z2_std_P,
            random_seed=rs
        )

        # Generate experimental data
        data_test, labels_test = Generate_De_target(
            sample_size=N_test,
            intercept_Y=intercept_Y,
            b_T_Y=b_T_Y,
            b_Z1_Y=b_Z1_Y,
            b_Z2_Y=b_Z2_Y,
            Z1_mean_PT=Z1_mean_PT,
            Z1_std_PT=Z1_std_PT,
            Z2_mean_PT=Z2_mean_PT,
            Z2_std_PT=Z2_std_PT,
            random_seed=rs
        )

        # Observational data coefficients
        intercept_T_obs = random_binary_coefficient()  # Intercept for T
        b_Z2_T_obs = random_continuous_coefficient()  # Strong effect of Z2 on T (confounder)

        # Generate observational data
        data_obs, labels_obs = Generate_Do_target(
            sample_size=No,
            intercept_Y=intercept_Y,
            b_T_Y=b_T_Y,
            b_Z1_Y=b_Z1_Y,
            b_Z2_Y=b_Z2_Y,
            intercept_T=intercept_T_obs,
            b_Z2_T=b_Z2_T_obs,
            Z1_mean_PT=Z1_mean_PT,
            Z1_std_PT=Z1_std_PT,
            Z2_mean_PT=Z2_mean_PT,
            Z2_std_PT=Z2_std_PT,
            random_seed=rs
        )

        print('how many 1 in observational', np.count_nonzero(labels_obs == 1))
        print('how many 1 in experimental', np.count_nonzero(labels_exp == 1))

        """Let's assume that Z1 is a latent and that MB(Y)=(T,Z2)"""
        P_Hzc = {}
        P_Hz_not_c = {}
        P_HZ = {}
        P_HZC = {}

        correct_IMB = 0
        num_samples = 1000
        found = False

        """Searching in these subsets of MB"""
        MB = tuple(range(data_obs.shape[1]))
        fixed_covariates = {0}
        initial_set = {0}

        covariates = set(MB)

        covariates = set(MB)

        Z = initial_set | fixed_covariates
        S, P_HZ, P_HZC = score_func(tuple(sorted(Z)))
        print(f"Initial set: {Z}, Score: {S:.4f}")
        found = False

        while not found:
            best_Z = Z
            best_S = S
            improved = False

            # Forward step
            print("\nTrying to add variables:")
            for cov in covariates - Z:
                new_Z = Z | {cov}
                new_S, P_HZ, P_HZC = score_func(tuple(sorted(new_Z)))
                print(f"  Try adding {cov}: new set {new_Z}, Score: {new_S:.4f}")
                if new_S > best_S + epsilon:
                    best_Z = new_Z
                    best_S = new_S
                    improved = True
                    print(f"    -> Improvement! Keeping new set {best_Z} with score {best_S:}")

            # Backward step (do not remove fixed covariates)
            print("\nTrying to remove variables:")
            for cov in Z - fixed_covariates:
                new_Z = Z - {cov}
                new_S, P_HZ, P_HZC = score_func(tuple(sorted(new_Z)))
                print(f"  Try removing {cov}: new set {new_Z}, Score: {new_S:.4f}")
                if new_S > best_S + epsilon:
                    best_Z = new_Z
                    best_S = new_S
                    improved = True
                    print(f"    -> Improvement! Keeping new set {best_Z} with score {best_S:}")

            # Check convergence
            if not improved:
                print("\nNo improvement found. Search has converged.")
                found = True
            else:
                Z = best_Z
                S = best_S
                print(f"\nUpdating current set to: {Z}, Score: {S:}")
        print('best_Z', Z)
        print('best_S', S)
        best_P_HZ = P_HZ[tuple(Z)]
        t = 0.5

        if best_P_HZ > t:
            print('best_P_HZ:',best_P_HZ)

            reg_variables = tuple(Z)

            # Find the set with the maximum P_Hz
            max_P_HZ = tuple(Z)

            print('Set with the max P_Hz from FindsABS', Z)

            list_of_Hz = {}

            for key, value in P_HZ.items():
                key_str = ''.join(str(i) for i in key)
                var_name = f'list_of_Hz_{key_str}'

                if var_name not in list_of_Hz:
                    list_of_Hz[var_name] = []

                list_of_Hz[var_name].append(value)

            """Only for the maximal set of s-admissible adjustment set calculate the performance based on log loss.
             Here: (0, 1, 2)"""
            """Use the hole set (0,1,2) as conditioning set when using onlt De or DO but for FindsAAS use the set with the
                        maximum P_HZ"""
            test_sub_data = data_test[:, MB]
            sub_data = data_obs[:, MB]
            exp_sub_data = data_exp[:, MB]

            test_sub_data_alg = data_test[:, max_P_HZ]
            sub_data_alg = data_obs[:, max_P_HZ]
            exp_sub_data_alg = data_exp[:, max_P_HZ]

            # Calclulate P_hat(Y) without multiply them with P(HZ|De,Do)
            """1. Assuming Hzc we can use both observational and experimental data"""
            rng_key = random.PRNGKey(np.random.randint(100))
            rng_key, rng_key_ = random.split(rng_key)
            combined_X = np.concatenate((sub_data_alg, exp_sub_data_alg), axis=0)  # Concatenate row-wise
            combined_y = np.concatenate((labels_obs, labels_exp), axis=0)  # Concatenate row-wise
            kernel = NUTS(binary_logistic_regression)
            num_samples = 1000
            mcmc = MCMC(kernel, num_warmup=1000, num_chains=1, num_samples=num_samples)
            mcmc.run(
                rng_key_, combined_X, combined_y
            )

            trace = mcmc.get_samples()
            intercept_alg = trace['alpha'].mean()
            slope_list = []
            for i in range(len(trace['beta'][0, :])):
                slope_list.append(trace['beta'][:, i].mean())

            slope_alg = np.array(slope_list)
            print('intercept and slope from observational + experimental:', intercept_alg, slope_alg)
            logit_alg = intercept_alg + np.dot(test_sub_data_alg, slope_alg)
            pr_1_alg = 1 / (1 + np.exp(-logit_alg))
            P_hat_alg = pr_1_alg
            P_HAT_alg[max_P_HZ] = P_hat_alg

            """1. Assuming Hzc_ we can use only experimental data"""
            rng_key = random.PRNGKey(np.random.randint(100))
            rng_key, rng_key_ = random.split(rng_key)
            kernel = NUTS(binary_logistic_regression)
            num_samples = 1000
            mcmc = MCMC(kernel, num_warmup=1000, num_chains=1, num_samples=num_samples)
            mcmc.run(
                rng_key_, exp_sub_data, labels_exp
            )

            trace = mcmc.get_samples()
            intercept_exp = trace['alpha'].mean()
            slope_list = []
            for i in range(len(trace['beta'][0, :])):
                slope_list.append(trace['beta'][:, i].mean())

            slope_exp = np.array(slope_list)
            print('intercept and slope from experimental:', intercept_exp, slope_exp)
            logit_exp = intercept_exp + np.dot(test_sub_data, slope_exp)
            pr_1_exp = 1 / (1 + np.exp(-logit_exp))
            pr_0_exp = 1 - pr_1_exp
            P_hat_exp = pr_1_exp
            P_HAT_exp[MB] = P_hat_exp

            """1. Assuming Hzc we can use only observational data"""
            rng_key = random.PRNGKey(np.random.randint(100))
            rng_key, rng_key_ = random.split(rng_key)
            kernel = NUTS(binary_logistic_regression)
            num_samples = 1000
            mcmc = MCMC(kernel, num_warmup=1000, num_chains=1, num_samples=num_samples)
            mcmc.run(
                rng_key_, sub_data, labels_obs
            )

            trace = mcmc.get_samples()
            intercept_obs = trace['alpha'].mean()
            slope_list = []
            for i in range(len(trace['beta'][0, :])):
                slope_list.append(trace['beta'][:, i].mean())

            slope_obs = np.array(slope_list)
            print('intercept and slope from observational:', intercept_obs, slope_obs)
            logit_obs = intercept_obs + np.dot(test_sub_data, slope_obs)
            pr_1_obs = 1 / (1 + np.exp(-logit_obs))
            pr_0_obs = 1 - pr_1_obs
            # how many 1s in test dataset
            P_1_prior = np.count_nonzero(labels_test == 1) / len(labels_test)
            P_hat_obs = pr_1_obs
            P_HAT_obs[MB] = P_hat_obs

            # Calclulate P_hat(Y) without multiply them with P(HZ|De,Do)
            """1. Assuming Hzc we can use both observational and experimental data and the hole (X,Z,W)"""
            rng_key = random.PRNGKey(np.random.randint(100))
            rng_key, rng_key_ = random.split(rng_key)
            combined_X = np.concatenate((sub_data, exp_sub_data), axis=0)  # Concatenate row-wise
            combined_y = np.concatenate((labels_obs, labels_exp), axis=0)  # Concatenate row-wise
            kernel = NUTS(binary_logistic_regression)
            num_samples = 1000
            mcmc = MCMC(kernel, num_warmup=1000, num_chains=1, num_samples=num_samples)
            mcmc.run(
                rng_key_, combined_X, combined_y
            )

            trace = mcmc.get_samples()
            intercept_all = trace['alpha'].mean()
            slope_list = []
            for i in range(len(trace['beta'][0, :])):
                slope_list.append(trace['beta'][:, i].mean())

            slope_all = np.array(slope_list)
            print('intercept and slope from observational + experimental:', intercept_all, slope_all)
            logit_all = intercept_all + np.dot(test_sub_data, slope_all)
            pr_1_all = 1 / (1 + np.exp(-logit_all))
            P_hat_all = pr_1_all
            P_HAT_all[MB] = P_hat_all

            # Epistrefw log loss mono gia ta cases pou exv ermineusei pws einai s-admissible-adjustment
            # if P_HZ[(0, 1, 2)] > P_HZC[(0, 1, 2)]:  # this means that is an adjustment set and also s-admissible
            """Use both Do and De"""
            P_Y_alg = P_HAT_alg[max_P_HZ]
            Log_loss_alg.append(log_loss(labels_test, P_Y_alg))
            print('Log loss function for our algorithm', log_loss(labels_test, P_Y_alg))

            P_Y_exp = P_HAT_exp[MB]
            Log_loss_exp.append(log_loss(labels_test, P_Y_exp))
            print('Log loss function for experimental data', log_loss(labels_test, P_Y_exp))

            P_Y_obs = P_HAT_obs[MB]
            Log_loss_obs.append(log_loss(labels_test, P_Y_obs))
            print('Log loss function for observational data', log_loss(labels_test, P_Y_obs))

            P_Y_all = P_HAT_all[MB]
            Log_loss_all.append(log_loss(labels_test, P_Y_all))
            print('Log loss function for ALL data', log_loss(labels_test, P_Y_all))

        print('Log loss from our algorithm', Log_loss_alg)
        print('Log loss in experimental', Log_loss_exp)
        print('Log loss in observational', Log_loss_obs)

        # Create a DataFrame
        AUC_results = {
            ''.join(str(i) for i in key): [value]
            for key, value in P_HZ.items()
        }
        print(AUC_results)
        Exp_results = pd.DataFrame({
            'log_loss_alg': Log_loss_alg,
            'log_loss_exp': Log_loss_exp,
            'Log_loss_obs': Log_loss_obs
        })

        # Save the DataFrame to a CSV file
        # AUC_results.to_csv(f'AUC_random_complete_results_N{Ne}.csv', index=False)
        # Exp_results.to_csv(f'experiments_random_complete_results_N{Ne}.csv', index=False)
