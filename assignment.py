import numpy as np

# settings
N_COLORS = 5
N_SHAPES = 5
N_TASKS = 2

N_TRIALS_TRAIN = 10000
N_TRIALS_TEST = 2000

N_CUE = 5
N_STIM = 5
N_DELAY = 5

SEQ_LEN = N_CUE + N_STIM + N_DELAY
INPUT_SIZE = N_COLORS + N_SHAPES + N_TASKS


def generate_dataset(n_trials):

    X = np.zeros(
        (n_trials, SEQ_LEN, INPUT_SIZE),
        dtype=np.float32
    )

    Y = np.zeros(
        n_trials,
        dtype=np.int64
    )

    # half color task, half shape task
    tasks = np.array(
        [0] * (n_trials // 2)
        + [1] * (n_trials // 2)
    )

    np.random.shuffle(tasks)


    for i in range(n_trials):

        # randomly choose color and shape
        c_idx = np.random.randint(N_COLORS)
        s_idx = np.random.randint(N_SHAPES)

        # balanced task
        t_idx = tasks[i]


        # task cue
        X[i, :, 10 + t_idx] = 1.0


        # stimulus period
        stim_start = N_CUE
        stim_end = N_CUE + N_STIM

        X[i, stim_start:stim_end, c_idx] = 1.0

        X[
            i,
            stim_start:stim_end,
            N_COLORS + s_idx
        ] = 1.0


        # target
        if t_idx == 0:
            Y[i] = c_idx
        else:
            Y[i] = s_idx


    return X, Y