    def Generate_DE_source(sample_size, intercept_Y, b_T_Y, b_Z2_Y, b_Z1_Y, Z1_mean_P, Z1_std_P, Z2_mean_P, Z2_std_P, rng):
      
        e = rng.normal(size=sample_size)

        # Generate independent variables
        Z1 = rng.normal(Z1_mean_P, Z1_std_P, sample_size)
        Z2 = rng.normal(Z2_mean_P, Z2_std_P, sample_size)
        T = rng.binomial(1, 0.5, sample_size)  # T ~ Bernoulli(0.5)

        # Generate Y using logistic model
        logit_Y = intercept_Y + b_T_Y * T + b_Z2_Y * Z2 + b_Z1_Y * Z1 + e
        pr_Y = np.clip(1 / (1 + np.exp(-logit_Y)), 1e-6, 1 - 1e-6)
        y_exp = rng.binomial(1, pr_Y)

        # x_exp = Z1.reshape(-1, 1)
        x_exp = np.column_stack((Z1, Z2))
        s_exp = np.zeros(sample_size)
        t_exp = T

        return x_exp, y_exp, t_exp, s_exp

    def Generate_Do_target(sample_size, intercept_Y, b_T_Y, b_Z2_Y, b_Z1_Y, intercept_T, b_Z2_T, Z1_mean_PT, Z1_std_PT, Z2_mean_PT, Z2_std_PT, rng):
      
        e = rng.normal(size=sample_size)

        # Generate confounders
        Z1 = rng.normal(Z1_mean_PT, Z1_std_PT, sample_size)
        Z2 = rng.normal(Z2_mean_PT, Z2_std_PT, sample_size)

        # Generate treatment
        logit_T = intercept_T + b_Z2_T * Z2
        prob_T = np.clip(1 / (1 + np.exp(-logit_T)), 1e-6, 1 - 1e-6)
        T = rng.binomial(1, prob_T)

        # Generate outcome
        logit_Y = intercept_Y + b_T_Y * T + b_Z2_Y * Z2 + b_Z1_Y * Z1 + e
        pr_Y = np.clip(1 / (1 + np.exp(-logit_Y)), 1e-6, 1 - 1e-6)
        y_obs = rng.binomial(1, pr_Y)

        # x_obs = Z1.reshape(-1, 1)
        x_obs = np.column_stack((Z1, Z2))
        s_obs = np.ones(sample_size)
        t_obs = T

        return x_obs, y_obs, t_obs, s_obs

    def random_binary_coefficient(rng):
        ranges = [(-2.5, -0.5), (0.5, 2.5)]
        chosen_range = ranges[rng.integers(len(ranges))]
        return rng.uniform(*chosen_range)

    def random_continuous_coefficient(rng):
        ranges = [(-1, -0.2), (0.2, 1)]
        chosen_range = ranges[rng.integers(len(ranges))]
        return rng.uniform(*chosen_range)

    def generate_random_params(rng):
        mean1 = rng.uniform(0, 10)
        std1 = rng.uniform(0, 10)

        mean2 = mean1
        while mean2 == mean1:
            mean2 = rng.uniform(0, 10)

        std2 = std1
        while std2 == std1:
            std2 = rng.uniform(0, 10)

        return mean1, std1, mean2, std2

    # Run the functions with different seeds
    
    n_runs = 100
    master_seed = 42  
    ss = np.random.SeedSequence(master_seed)
    child_seeds = ss.spawn(n_runs)
    # Create RNGs for each run
    rngs = [np.random.default_rng(s) for s in child_seeds]

    for k in range(n_runs):

        rng = rngs[k]

        intercept_Y = random_binary_coefficient(rng)  # Intercept for Y
        b_T_Y = random_binary_coefficient(rng)  # Effect of T on Y
        b_Z2_Y = random_continuous_coefficient(rng)  # Effect of Z2 on Y
        b_Z1_Y = random_continuous_coefficient(rng)  # Effect of Z1 on Y
        Z1_mean_P, Z1_std_P, Z1_mean_PT, Z1_std_PT = generate_random_params(rng)
        Z2_mean_P, Z2_std_P, Z2_mean_PT, Z2_std_PT = generate_random_params(rng)

        # Generate experimental data
        x_rct, y_rct, t_rct, s_rct = Generate_DE_source(
            sample_size=Ne,
            intercept_Y=intercept_Y,
            b_T_Y=b_T_Y,
            b_Z1_Y=b_Z1_Y,
            b_Z2_Y=b_Z2_Y,
            Z1_mean_P=Z1_mean_P,
            Z1_std_P=Z1_std_P,
            Z2_mean_P=Z2_mean_P,
            Z2_std_P=Z2_std_P,
            rng=rng
        )

        # Observational data coefficients
        intercept_T_obs = random_binary_coefficient(rng)  # Intercept for T
        b_Z2_T_obs = random_continuous_coefficient(rng)  # Effect of Z2 on T (confounder)

        # Generate observational data
        x_obs, y_obs, t_obs, s_obs = Generate_Do_target(
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
            rng=rng
        )

        x = np.vstack((x_rct, x_obs))
        x = x[:, [0]]
        y = np.hstack((y_rct, y_obs))
        t = np.hstack((t_rct, t_obs))
        s = np.hstack((s_rct, s_obs))
