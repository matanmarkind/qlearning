#import matplotlib
#matplotlib.use('Agg')  # Can't show using this setup.
from GridworldQnet import GridworldQnet
from CleanProcessGroup import CleanProcessGroup
import gridworld

from collections import defaultdict
from subprocess import Popen
from datetime import datetime
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import os, socket, time, argparse, sys, tempfile, glob, re, resource

#plt = matplotlib.pyplot

"""
This file is focused on training identical Neural Nets in separate processes.
This is to utilize multi core architecture.
1) Starts numerous processes to train identical NN
2) When that finishes combines the resultant network
3) Repeat, now using the combined as the initial checkpoint for each process.
"""
DEFAULT_EPISODES = 100000

# Delimeter used in model names to separate between
# scope and episodes. Models should be named
# model-<scope>-<episodes>.ckpt
DELIM = '-'

parser = argparse.ArgumentParser()

# Arguments for single and multi training
parser.add_argument('--mode', type=str,
                    help="(train, multitrain, show, test)")
parser.add_argument('--model_dir', type=str, default='',
                    help="Directory to save models to if (multi)training, or "
                         " to load ckpts from if combining. Not required to "
                         " show or test.")
parser.add_argument('--restore_model', type=str, default='',
                    help="Checkpoint to load base model from.")
parser.add_argument('--update_target_net_rate', type=int, default=200,
                    help="Number of episodes between copying the main net to the target net for DoubleDQN")
parser.add_argument('--episodes', type=int, default=DEFAULT_EPISODES,
                    help="Number of games to play.")
parser.add_argument('--game_len', type=int, default=50,
                    help="Number of turns per episode.")
parser.add_argument('--experiences_size', type=int, default=10000,
                    help="Experience buffer size. Doesn't start training until this is full.")
parser.add_argument('--batch_size', type=int, default=8,
                    help="Number of experiences to batch together for processing (higher for GPU's)")
parser.add_argument('--replays', type=int, default=8,
                    help="Approximate number of times to replay each experience "
                         "(avg, not actually per experience)")
parser.add_argument('--discount', type=float, default=.99,
                    help="Discount factor on Q values from future states")
parser.add_argument('--random_decay', type=float, default=.999,
                    help="The exponential rate at which to decay random action selection.")
parser.add_argument('--learning_rate', type=float, default=.00025,
                    help="Learning rate for Gradient Optimizer.")

# Multi trianing args
parser.add_argument('--multitrain_random_action', type=float, default=.999,
                    help="Initial chance of performing a random action."
                         "Reduces exponentially across multi training.")
parser.add_argument('--num_trainers', type=int, default=4,
                    help="Number of subprocesses to use for training.")

# Regular training args
parser.add_argument('--random_action', type=float, default=.999,
                    help='Initial chance of performaing a random action.'
                         'Reduces exponentially with random_decay.')
parser.add_argument('--ckpt_every', type=int, default=DEFAULT_EPISODES//10,
                    help="How often print results and save a new checkpoint")


# Helper functions
def get_hostname() -> str:
  """
  Get the stripped down version of the host name to be used for
  sudmodel_dir, ckpt names, scoping.
  Names had '.' and '-' in them which I want for scoping/file suffix.
  keeps only letters, numbers, _.
  """
  return re.sub('\W+', '', socket.gethostname())

def get_scope(model_path: str) -> str:
    """
    Calculates the scope a model's contents are in. Assumes files have the
    pattern <XXX>-<scope>.*
    Aka that there is a '-' delimeter and the scope is in index 1.
    :param model_path: path to a tensorflow checkpoint. This doesn't have to be
    to an actual file since TF checkpoints involve multiple files.
    :return: scope that the model in this file are saved in.
    """
    fname = os.path.basename(model_path)
    return fname.split('.')[0].split(DELIM)[1]

def cur_scope() -> str:
    """
    Calculates the current scope to use
    :return:
    """
    return '_'.join((get_hostname(), str(os.getpid())))

def get_episodes(model_path: str) -> str:
    """
    Calculates the scope a model's contents are in. Assumes files have the
    pattern <XXX>-<XXX>-<episodes>
    Aka that there is a '-' delimeter and the scope is in index 2.
    :param model_path: path to a tensorflow checkpoint. This doesn't have to be
    to an actual file since TF checkpoints involve multiple files.
    :return: number of episodes checkpoint contains.
    """
    if model_path == '':
        return 0

    fname = os.path.basename(model_path)
    return int(fname.split('.')[0].split(DELIM)[2])

def get_net(scope, in_shape, n_actions, discount, learning_rate,
            experiences_size, batch_size, update_target_net_rate):
    """
    :return: A GridworldQnet created within a specific scope that can be reused.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        return GridworldQnet(in_shape, n_actions, discount, learning_rate,
                             experiences_size, batch_size,
                             update_target_net_rate, tf.train.AdamOptimizer)

def get_dummy_net(scope, in_shape, n_actions):
    """
    Useful if you don't want to train a GridworldQnet. For example if you
    just want to load up a preexisting model and test it without doing
    any learning.
    :return: A GridworldQnet created within a specific scope that can be reused,
        but won't learn.
    """
    return get_net(scope, in_shape, n_actions, 0, 0, 0, 0, 0)

def restore_from_ckpt(sess, target_scope, ckpt_path, img_shape, n_actions):
    """
    Restore a model from model_path into target_scope. Assumes scoping rules
    found in get_scope & that the target model has been created already. If
    model_path is empty this won't do anything.
    :param sess: tf.Session
    :param target_scope: scope to copy restored model into.
    :param ckpt_path: fpath to checkpoint
    :param img_shape: shape of input
    :param n_actions: number of possible outputs
    :return: nothing
    """
    if ckpt_path == '':
        #TODO: Think about if want this to be the callers responsibility.
        return

    loaded_scope = get_scope(ckpt_path)
    loaded_qnet = get_dummy_net(loaded_scope, img_shape, n_actions)
    loaded_saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=loaded_scope))
    loaded_saver.restore(sess, ckpt_path)

    loaded_vars = tf.trainable_variables(loaded_scope)
    target_vars = tf.trainable_variables(target_scope)
    if target_scope == loaded_scope:
        # In this case the restore would have placed the values into
        # the target automatically.
        return

    loaded_vars = tf.trainable_variables(loaded_scope)
    target_vars = tf.trainable_variables(target_scope)
    for target, loaded in zip(target_vars, loaded_vars):
        target.assign(loaded.value())

def make_ckpt_path(direc, scope, eps):
    """
    Creates the fpath for the checkpoint according the naming rules
    we use.
    """
    return os.path.join(
            direc,
            DELIM.join(('model', scope, str(eps))) + '.ckpt')

def get_models_to_combine(models_dir):
    """
    Get the models that we are going to load up to combine. Only takes
    the checkpoint with the highest number of episodes from each scope. Each
    checkpoint should be named. Only meant for usage
    model-<scope>-<episodes>.ckpt
    :return: list of checkpoints to combine together
    """

    # Get a list of the models. Each checkpoint generates 3 files so the inner
    # set comprehension is to get 1 name per checkpoint instead of all 3 files.
    # The outer list comprehension
    # is to ignore other files (like "checkpoint") that we aren't interested in.
    ckpts = [j for j in {i.split('.')[0] for i in os.listdir(models_dir)}
             if j.startswith('model')]

    # Only take the checkpoint from a given scope with most episodes.
    per_scope_ckpts = defaultdict(list)
    for c in ckpts:
        per_scope_ckpts[get_scope(c)].append(c)
    maxes, maxes_i = defaultdict(int), defaultdict(int)
    for scope, checkpoints in per_scope_ckpts.items():
        for i, checkpoint in enumerate(checkpoints):
            episodes = int(checkpoint.split(DELIM)[2])
            if episodes > maxes[scope]:
                maxes[scope] = episodes
                maxes_i[scope] = i

    return [checkpoints[maxes_i[scope]] for scope, checkpoints in
            per_scope_ckpts.items()]


def combine_nets(combine_dir, submodels_dir, episodes=0):
    """
    Combines a set of Neural Networks by taking the avg value for each
    neuron, weighted by the number of experiences.
    :param combine_dir: directory to save combined model to
    :param submodel_dir: directory to load models from
    :param episodes: base number of episodes to add to model total. Ends up being the total count Ends up being the total count
    :return: path to combined model
    """
    avg_scope = cur_scope()
    env = gridworld.gameEnv(partial=False, size=5)
    tf.reset_default_graph()

    # create the network that will be used to combine the others.
    avg_qnet = get_dummy_net(avg_scope, env.state.shape, env.actions)
    avg_vars = tf.trainable_variables(avg_scope)

    init = tf.global_variables_initializer()
    avg_saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=avg_scope))
    models = get_models_to_combine(submodels_dir)
    with tf.Session() as sess:
        sess.run(init)

        for i, m in enumerate(models):
            _, m_scope, eps = m.split(DELIM)
            eps = int(eps)
            m_qnet = get_dummy_net(m_scope, env.state.shape, env.actions)
            restore_from_ckpt(sess, m_scope,
                              os.path.join(submodels_dir, m + '.ckpt'),
                              env.state.shape, env.actions)
            m_vars = tf.trainable_variables(m_scope)
            for avg, loaded in zip(avg_vars, m_vars):
                sess.run(
                    avg.assign(
                        avg.value() * episodes / (episodes + eps)
                        + loaded.value() * eps / (episodes + eps)))
            episodes += eps

        # Save the combined NN
        ckpt_path = make_ckpt_path(combine_dir, avg_scope, episodes)
        avg_saver.save(sess, ckpt_path)
    return ckpt_path


def clean_dir(direc):
    """
    Cleans out all files from a directory if it exists. Only cleans
    files, doesn't recursively inspect subdirectories.
    """
    for fpath in glob.glob(os.path.join(direc, '*')):
        if os.path.isfile(fpath):
            os.remove(fpath)

def remove_ckpt(ckpt_path):
    """
    Deletes all ckpt files associates with a ckpt in a given directory.
    :param ckpt_path: path to ckpt. Something like 
        ~/my.project/ckpts/model-scope-eps.ckpt
    """
    ckpt = os.path.basename(ckpt_path)
    ckpt_dir = os.path.dirname(ckpt_path)
    files = [f for f in os.listdir(ckpt_dir)
             if os.path.isfile(f) and f.startswith(ckpt)]
    for f in files:
        os.remove(os.path.join(ckpt_dir, f))

def normalize_img(img):
    """
    Take in the state (image) and noralize the date for the NN.
    0 centered, range of -1 to 1.
    :param state: RGB image
    :return:
    """
    return (img / 128. - 1).astype(np.float32)


# Functions that represent tasks for this script to perform.


def train(args):
    """
    Create a GridworldQnet and have it play games and train.
    """
    scope = cur_scope()
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    e = args.random_action
    env = gridworld.gameEnv(partial=False, size=5)
    tf.reset_default_graph()

    # Convert update_target_net_rate from number of games between copies to
    # the number of 'updates' between.
    update_target_net_rate = \
        args.update_target_net_rate * args.game_len * args.replays
    qnet = get_net(
        scope, env.state.shape, env.actions, args.discount,
        args.learning_rate, args.experiences_size, args.batch_size,
        update_target_net_rate)

    rewards = []

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        restore_from_ckpt(sess, scope, args.restore_model, env.state.shape,
                           env.actions)

        for ep in range(args.episodes):
            state = normalize_img(env.reset())
            done = False
            turn = 0
            net_reward = 0
            n_updates = 0
            while not done and turn < args.game_len:
                action = qnet.predict(sess, np.array([state]))[0]
                if np.random.rand(1) < e:
                    action = qnet.rand_action()

                next_state, reward, done = env.step(action)
                next_state = normalize_img(next_state)
                qnet.add_experience(state, action, reward, next_state, done)
                for i in range(abs(int(reward))*10):
                    # Add lots of examples when there is a reward, cuz sparse
                    qnet.add_experience(state, action, reward, next_state, done)

                turn += 1
                state = next_state
                net_reward += reward

                while qnet.experiences_full and n_updates < (turn * args.replays):
                    # This is where the actual training happens. We look back
                    # back and sample experiences we have had and learn by
                    # replaying them.
                    n_updates += args.batch_size
                    qnet.update(sess)

            e *= args.random_decay
            rewards.append(net_reward)

            if (ep + 1) % args.ckpt_every == 0 and ep > 0:
                ckpt_path = make_ckpt_path(args.model_dir, scope, ep + 1)
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S "),
                      os.path.basename(ckpt_path), ' ',
                      resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//2**10,
                      'kB random_action={:.5f}'.format(e), ' game_len=', args.game_len,
                      ' Net reward last ', args.ckpt_every, ' games: ',
                      sum(rewards[-args.ckpt_every:]),
                      sep='')
                saver.save(sess, ckpt_path)

def show_game(ckpt, game_len):
    """
    loads a model up from ckpt and plays a single game to show.
    """
    scope = cur_scope()
    env = gridworld.gameEnv(partial=False, size=5)
    tf.reset_default_graph()

    qnet = get_dummy_net(scope, env.state.shape, env.actions)
    plt.ion()
    plt.show()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        restore_from_ckpt(sess, scope, ckpt, env.state.shape, env.actions)
        state = env.reset()
        for _ in range(game_len):
            plt.imshow(state)
            plt.draw()
            plt.pause(.1)
            action = qnet.predict(sess, np.array([state]))[0]
            state, _, _ = env.step(action)

    plt.close()
    plt.ioff()

def test_model(ckpt, num_games, game_len):
    """
    Loads up a model and plays a number of games with it. Doesn't do
    any training, but reports the net reward at the end.
    """
    scope = cur_scope()
    env = gridworld.gameEnv(partial=False, size=5)
    tf.reset_default_graph()

    qnet = get_dummy_net(scope, env.state.shape, env.actions)
    net_reward = 0
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        restore_from_ckpt(sess, scope, ckpt, env.state.shape,
                           env.actions)
        for _ in range(num_games):
            state = env.reset()
            for i in range(game_len):
                action = qnet.predict(sess, np.array([state]))[0]
                state, reward, _ = env.step(action)
                net_reward += reward

    print('num_games =', num_games,
          '    game_len =', game_len,
          '    score =', net_reward,
          '    ckpt =', ckpt)

def multitrain(args):
    """
    Runs a group of processes that will each train a GridworldQnet.
    When each training ends, combines the models together.
    Then retrains starting from the combined model.
    Continues forever until forcibly stopped.
    """
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    restore_model = args.restore_model
    eps = get_episodes(restore_model)
    submodels_dir = os.path.join(args.model_dir, get_hostname())

    while True:
        with CleanProcessGroup(foreground=True):
            clean_dir(submodels_dir)

            # Start processess that will do training
            # TODO: maybe instead of just saving at the end of each multitraining round
            # we can save more frequently, and just delete any previous checkpoint from
            # the same scope so there will only be one, but will be saved more often
            # so interrupts don't lose as much. (only keep last 1? something like this
            # exists for the higher level TF things)
            cmd = ("python3",
                   "gridworld_trainer.py",  # Assumes I am in the same dir.
                   "--mode=train",
                   "--model_dir=" + submodels_dir,
                   "--restore_model=" + restore_model,
                   "--update_target_net_rate=" + str(args.update_target_net_rate),
                   "--episodes=" + str(args.episodes),
                   "--ckpt_every=" + str(args.episodes),  # Only want a checkpoint at the end
                   "--game_len=" + str(args.game_len),
                   "--experiences_size=" + str(args.experiences_size),
                   "--batch_size="+ str(args.batch_size),
                   "--replays=" + str(args.replays),
                   "--discount=" + str(args.discount),
                   "--random_action=" + str(args.multitrain_random_action ** (eps+1)),
                   "--random_decay=" + str(args.random_decay),
                   "--learning_rate=" + str(args.learning_rate))
            trainers = [Popen(cmd) for i in range(args.num_trainers)]

            # Display current model
            show_game(restore_model, args.game_len)
            # Play a number of games to show net reward changes
            test_model(restore_model, args.episodes, args.game_len)

            while None in [sp.poll() for sp in trainers]:
                # Check once a minute to see if training is done
                time.sleep(60)

        # Combine the nets together.
        if restore_model != '':
            remove_ckpt(restore_model)
        restore_model = combine_nets(args.model_dir, submodels_dir, eps)
        eps = get_episodes(restore_model)

def combine(model_dir):
    """
    Combines the checkpoints in model_dir together. This is meant to be the
    way to combine models from separate machines (so within GridworldModels).

    Removes all files from model_dir only leaving the combined ckpt.
    """
    with tempfile.TemporaryDirectory() as tdir:
        ckpt = combine_nets(tdir, model_dir)
        clean_dir(model_dir)
        for f in os.listdir(tdir):
            os.rename(f, os.path.join(model_dir, os.path.basename(f)))

def main():
    args = parser.parse_args(sys.argv[1:])
    
    if args.mode == 'train':
        assert args.model_dir != '', 'Must provide a directory to save ckpts to'
        train(args)
    elif args.mode == 'show':
        show_game(args.restore_model, args.game_len)
    elif args.mode == 'test':
        test_model(args.restore_model, args.episodes, args.game_len)
    elif args.mode == 'multitrain':
        assert args.model_dir != '', 'Must provide a directory to save ckpts to (./GridworldModels)'
        multitrain(args)
    elif args.mode == 'combine':
        assert args.model_dir != '', 'Must provide a directory to load ckpts from/to'
        combine(args.model_dir)
    else:
        print('No valid mode provided.')


if __name__ == '__main__':
    main()

