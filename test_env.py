from ToyGridEnv import ToyGridEnv

env = ToyGridEnv(seed=10, max_episode_length=50, action_repeat=1)

# state = env.reset()
# asdf = [np.copy(state[0:2])]
# for _ in range(100):
#     act = np.random.uniform(-1,1,2)
#     state, reward, done, info = env.step(act)
#     asdf.append(state[0:2].copy())
# env.render_path(path_list=[asdf])

act_step = 2
path = []
for _ in range(100):
    act = env.sample_random_action()
    # act = np.random.uniform(-1,1,2)
    # state, reward, done, info = env.step(act)
    state, reward, done = env.step(act)
    print(reward)
    path.append(state[0][:2].detach().numpy().copy())

env.render_path([path])

