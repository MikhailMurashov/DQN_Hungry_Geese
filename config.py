min_frames_for_train = 2
buffer_size = 50000
num_iterations = 100
save_target_model = 500

num_frames = 2
num_actions = 4
geese_shape = (num_frames, 11, 7)

batch_size = 1
learning_rate = 0.00025

initial_epsilon = 1.0
final_epsilon = 0.05
final_iteration_num = num_iterations / 5

gamma = 0.99  # decay rate of past observations
