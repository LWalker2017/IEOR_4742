import tensorflow as tf
import numpy as np
import getopt
import shutil
import pickle
import sys
import os

import memory_operations as op

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))

def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    index = int(index)
    vec[index] = 1.0
    return vec

def prepare_sample(sample, target_code, word_space_size):
    
    """
    prepares the input/output sequence of a sample story by encoding it
    into one-hot vectors and generates the necessary loss weights
    """
    
    input_vec = np.array(sample[0]['inputs'], dtype=np.float32)
    output_vec = np.array(sample[0]['inputs'], dtype=np.float32)
    
    seq_len = input_vec.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)

    target_mask = (input_vec == target_code)
    output_vec[target_mask] = sample[0]['outputs']
    weights_vec[target_mask] = 1.0

    input_vec = np.array([onehot(code, word_space_size) for code in input_vec])
    output_vec = np.array([onehot(code, word_space_size) for code in output_vec])

    return (
        np.reshape(input_vec, (-1, word_space_size)),
        np.reshape(output_vec, (-1, word_space_size)),
        seq_len,
        np.reshape(weights_vec, (-1, 1))
    )

task_dir = os.path.dirname(os.path.realpath(__file__))

llprint("Loading Data ... ")
lexicon_dict = load(os.path.join(task_dir, "data\\en-10k\\lexicon-dict.pkl"))
data = load(os.path.join(task_dir, "data\\en-10k\\train\\train.pkl"))
llprint("Done!\n")

# the model parameters

# memory size NxW
N = 256
W = 64 

# number of read heads
R = 4     # memory parameters

X = 159   # input size
Y = 159   # output size

NN = 256  # controller's network output size

# setting up zeta
zeta_size = R*W + 3*W + 5*R + 3

# training parameters, number of iterations
iterations = 10000

# in case of using RMSProp
learning_rate = 0.001
momentum = 0.9


def network(step_input, state):
    
    """
    defines the recurrent neural network operation
    """
    
    global NN
    step_input = tf.expand_dims(step_input, 0)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(NN)

    return lstm_cell(step_input, state)

# START: Computational Graph
graph = tf.Graph()


with graph.as_default():
    
    # optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)

    # placeholders
    input_data = tf.placeholder(tf.float32, [None, X])
    target_output = tf.placeholder(tf.float32, [None, Y])
    loss_weights = tf.placeholder(tf.float32, [None, 1])
    sequence_length = tf.placeholder(tf.int32)

    initial_nn_state = tf.nn.rnn_cell.BasicLSTMCell(NN).zero_state(1, tf.float32)

    empty_unpacked_inputs = tf.TensorArray(tf.float32, sequence_length)
    unpacked_inputs = empty_unpacked_inputs.unstack(input_data)
    outputs_container = tf.TensorArray(tf.float32, sequence_length)  # accumulates the step outputs
    t = tf.constant(0, dtype=tf.int32)

    def step_op(time, memory_state, controller_state, inputs, outputs):
        
        """
        defines the operation of one step of the sequence
        """
        global N, W, R

        step_input = inputs.read(time)
        M, u, p, L, wr, ww, r = memory_state

        with tf.variable_scope('controller'):

            std = lambda input_size: np.minimum(0.01, np.sqrt(2./ input_size))
            
            Xt = tf.concat(values=[step_input, tf.reshape(r, [-1])],axis=0,)
            #Xt = tf.concat(0, [step_input, tf.reshape(r, [-1])])

            nn_output, nn_state = network(Xt, controller_state)

            W_zeta = tf.get_variable('W_zeta', [NN, zeta_size], tf.float32, tf.truncated_normal_initializer(stddev=std(NN)))
            #W_zeta = tf.get_variable("W_zeta", [NN, zeta_size], tf.truncated_normal_initializer(stddev=std(NN)))
            
            #weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
            #W = tf.get_variable("W", weight_shape, initializer=weight_init)

            W_y = tf.get_variable('W_y', [NN, Y], tf.float32, tf.truncated_normal_initializer(stddev=std(NN)))
            #W_y = tf.get_variable("W_y", [NN, Y], tf.truncated_normal_initializer(stddev=std(NN)))
            #W_y = tf.get_variable('W_y', [NN, Y], tf.float32, tf.truncated_normal_initializer(stddev=std(NN)))

            
            pre_output = tf.matmul(nn_output, W_y)
            zeta = tf.squeeze(tf.matmul(nn_output, W_zeta))
            kr, br, kw, bw, e, v, f, ga, gw, pi = op.parse_interface(zeta, N, W, R)
            
            # write head operations
            u_t = op.ut(u, f, wr, ww)

            #a_t = op.at(u_t, N)
            cw_t = op.C(M, kw, bw)
            a_t = op.at(u_t, N)
            ww_t = op.wwt(cw_t, a_t, gw, ga)
            M_t = op.Mt(M, ww_t, e, v)
            L_t = op.Lt(L, ww_t, p, N)
            p_t = op.pt(ww_t, p)

            # read heads operations
            cr_t = op.C(M_t, kr, br)
            wr_t = op.wrt(wr, L_t, cr_t, pi)
            r_t = op.rt(M_t, wr_t)

            W_r = tf.get_variable('W_r', [W*R, Y], tf.float32, tf.truncated_normal_initializer(stddev=std(W*R)))
            flat_rt = tf.reshape(r_t, [-1])
            final_output = pre_output + tf.matmul(tf.expand_dims(flat_rt, 0), W_r)
            updated_outputs = outputs.write(time, tf.squeeze(final_output))

            return time + 1, (M_t, u_t, p_t, L_t, wr_t, ww_t, r_t), nn_state, inputs, updated_outputs
    
    _, _, _, _, final_outputs = tf.while_loop(
        cond = lambda time, *_: time < sequence_length,
        body = step_op, loop_vars=(t, op.init_memory(N,W,R), initial_nn_state, unpacked_inputs, outputs_container),
        parallel_iterations=32,
        swap_memory=True
    )

    # pack the individual steps outputs into a single (sequence_length x Y) tensor
    packed_output = final_outputs.stack()
    loss = tf.reduce_mean(loss_weights*tf.nn.softmax_cross_entropy_with_logits(logits=packed_output, labels=target_output))
    gradients = optimizer.compute_gradients(loss)
    
    
    # clipping the gradients by value to avoid explosion
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_value(grad, -10, 10), var)
    apply_grads = optimizer.apply_gradients(gradients)
    # END: Computational Graph

    # Reading command line arguments and adapting parameters
    options,_ = getopt.getopt(sys.argv[1:], '', ['iterations='])
    for opt in options:
        iterations = int(opt[1])

    with tf.Session(graph=graph) as session:

        #session.run(tf.global_variables_initializer())
        session.run(tf.initializers.global_variables())

        last_100_losses = []
        print("")
        for i in range(iterations):

            llprint("\r iteration %d/%d" % (i, iterations))

            sample = np.random.choice(data, 1)
            input_seq, target_seq, seq_len, weights = prepare_sample(sample, lexicon_dict['-'], 159)

            loss_value,_, = session.run([loss, apply_grads], feed_dict={
                input_data: input_seq,
                target_output: target_seq,
                sequence_length: seq_len,
                loss_weights: weights
            })

            last_100_losses.append(loss_value)
            if i % 100 == 0:
                print("\n\tAvg. Cross-Entropy Loss: %0.6f" % (np.mean(last_100_losses)))
                last_100_losses = []

        model_path = os.path.join(task_dir, 'bAbI-model')
        
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.mkdir(model_path)
        
        tf.train.Saver().save(session, os.path.join(model_path, 'model.ckpt'))