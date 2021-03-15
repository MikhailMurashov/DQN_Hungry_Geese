import config


def get_epsilon_for_iteration(iter_num):
    if iter_num is 0:
        return config.initial_epsilon
    elif iter_num > config.final_iteration_num:
        return config.final_epsilon
    else:
        # return -0.0168 * iter_num**2 + 3.0867 * iter_num + 41.3745
        return -0.0001 * iter_num + 1.0001
