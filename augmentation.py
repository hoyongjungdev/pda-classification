import numpy as np

def jitter(x, y, n_jitter, scale):
    length = x.shape[1]
    dim = x.shape[2]

    js = np.random.choice(len(x), size=n_jitter)

    jittered_arr = np.empty((n_jitter, length, dim))
    answer_arr = np.empty(n_jitter)
    jitter_original_arr = np.empty((n_jitter, length, dim))

    for i in range(len(js)):
        ji = js[i]
        data = x[ji]

        j = np.random.normal(scale=scale, size=(length, dim))
        jittered = data + j

        jittered_arr[i] = jittered
        answer_arr[i] = y[ji]
        jitter_original_arr[i] = data

    return jittered_arr, answer_arr, jitter_original_arr

def scale(x, y, n_scale, scale):
    length = x.shape[1]
    dim = x.shape[2]

    ss = np.random.choice(len(x), size=n_scale)

    scaled_arr = np.empty((n_scale, length, dim))
    answer_arr = np.empty(n_scale)
    scale_original_arr = np.empty((n_scale, length, dim))

    for i in range(len(ss)):
        si = ss[i]
        data = x[si]

        s = np.random.normal(loc=1, scale=scale, size=(1, dim))
        scaled = data * s

        scaled_arr[i] = scaled
        answer_arr[i] = y[si]
        scale_original_arr[i] = data

    return scaled_arr, answer_arr, scale_original_arr

def augment_jitter(n_jitter, alpha, train_x, train_y):
    jittered_x, jittered_y, _ = jitter(train_x, train_y, n_jitter, alpha)

    ax = np.empty((train_x.shape[0] + n_jitter, train_x.shape[1], train_x.shape[2]))
    ay = np.empty(train_x.shape[0] + n_jitter)

    for i in range(len(train_x)):
        ax[i] = train_x[i]
        ay[i] = train_y[i]

    for i in range(n_jitter):
        ax[i+len(train_x)] = jittered_x[i]
        ay[i+len(train_x)] = jittered_y[i]

    return ax, ay
