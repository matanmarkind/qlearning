from gridworld_trainer import *

model_dir = './GridworldModels'
submodels_dir = os.path.join(model_dir, get_hostname())

avg_scope = cur_scope()
env = gridworld.gameEnv(partial=False, size=5)
episodes = 0  # Net episodes from checkpoints, used for weighting.
tf.reset_default_graph()

# create the network that will be used to combine the others.
avg_qnet = get_dummy_net(avg_scope, env.state.shape, env.actions)
avg_vars = tf.trainable_variables(avg_scope)
print('avg_vars =', len(avg_vars))

init = tf.global_variables_initializer()
avg_saver = tf.train.Saver(
    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=avg_scope))
models = get_models_to_combine(submodels_dir)
print('models =', models)

with tf.Session() as sess:
    sess.run(init)

    for i, m in enumerate(models):
        _, m_scope, eps = m.split(DELIM)
        eps = int(eps)
        m_qnet = get_dummy_net(m_scope, env.state.shape, env.actions)
	
	#TODO: remove this?
        m_vars = tf.trainable_variables(m_scope)
        print(m, 'm_vars =', len(m_vars))
        for avg, loaded in zip(avg_vars, m_vars):
            spaces = ''.join([' ' for i in range(60)])
            print('/'.join(avg.name.split('/')[1:]),
                  spaces[len(avg.name):],
                  '/'.join(loaded.name.split('/')[1:]))

        restore_from_ckpt(
            sess, m_scope, os.path.join(submodels_dir, m + '.ckpt'),
            env.state.shape, env.actions)
        m_vars = tf.trainable_variables(m_scope)
        print(m, 'm_vars =', len(m_vars))

        for avg, loaded in zip(avg_vars, m_vars):
            sess.run(
                avg.assign(
                    avg.value() * episodes / (episodes + eps)
                    + loaded.value() * eps / (episodes + eps)))
        episodes += eps

    # Save the combined NN
    avg_saver.save(
            sess,
            make_ckpt_path(model_dir, avg_scope, episodes))

