import math
import numpy as np
import torch

def iterate_batch_multiple_seq_minibatches(inputs, targets, batch_size, seq_length, shuffle_idx=None, augment_seq=False):
    """
    Generate a generator that return a batch of sequences of inputs and targets.

    This function randomly selects batches of multiple sequence. It then iterates
    through multiple sequence in parallel to generate a sequence of inputs and
    targets. It will append the input sequence with 0 and target with -1 when
    the lenght of each sequence is not equal.
    """
    # print(f"using batch_size{batch_size}, seq_length{seq_length}")
    assert len(inputs) == len(targets)
    n_inputs = len(inputs)

    if shuffle_idx is None:
        # No shuffle
        seq_idx = np.arange(n_inputs)
    else:
        # Shuffle subjects (get the shuffled indices from argument)
        seq_idx = shuffle_idx

    input_sample_shape = inputs[0].shape[1:]
    print("****input_sample_shape labels ", input_sample_shape)

    target_sample_shape = targets[0].shape[1:]

    # Compute the number of maximum loops
    n_loops = int(math.ceil(len(seq_idx) / batch_size))  # todo ?
    # For each batch of subjects (size=batch_size)
    for l in range(n_loops):
        start_idx = l*batch_size
        end_idx = (l+1)*batch_size
        seq_inputs = np.asarray(inputs)[seq_idx[start_idx:end_idx]]
        seq_targets = np.asarray(targets)[seq_idx[start_idx:end_idx]]

        if augment_seq:
            # Data augmentation: multiple sequences
            # Randomly skip some epochs at the beginning -> generate multiple sequence
            max_skips = 5
            for s_idx in range(len(seq_inputs)):
                n_skips = np.random.randint(max_skips)
                seq_inputs[s_idx] = seq_inputs[s_idx][n_skips:]
                seq_targets[s_idx] = seq_targets[s_idx][n_skips:]

        # Determine the maximum number of batch sequences
        n_max_seq_inputs = -1
        for s_idx, s in enumerate(seq_inputs):
            if len(s) > n_max_seq_inputs:
                n_max_seq_inputs = len(s)

        n_batch_seqs = int(math.ceil(n_max_seq_inputs / seq_length))  # 一批个体（15个），最多有n_batch_seqs个seq， 每个seq的长度是20（seq_length）

        # For each batch sequence (size=seq_length)
        for b in range(n_batch_seqs):
            start_loop = True if b == 0 else False
            start_idx = b*seq_length
            end_idx = (b+1)*seq_length
            batch_inputs = np.zeros((batch_size, seq_length) + input_sample_shape, dtype=np.float32)  # batch_size=15, seq_length=20
            batch_targets = np.zeros((batch_size, seq_length) + target_sample_shape, dtype=np.int64)
            batch_weights = np.zeros((batch_size, seq_length), dtype=np.float32)
            batch_seq_len = np.zeros(batch_size, dtype=np.int64)
            # For each subject
            for s_idx, s in enumerate(zip(seq_inputs, seq_targets)):
                # (seq_len, sample_shape)
                each_seq_inputs = s[0][start_idx:end_idx]
                each_seq_targets = s[1][start_idx:end_idx]
                batch_inputs[s_idx, :len(each_seq_inputs)] = each_seq_inputs
                batch_targets[s_idx, :len(each_seq_targets)] = each_seq_targets
                batch_weights[s_idx, :len(each_seq_inputs)] = 1
                batch_seq_len[s_idx] = len(each_seq_inputs)
            batch_x = batch_inputs.reshape((-1,) + input_sample_shape)
            batch_y = batch_targets.reshape((-1,) + target_sample_shape)
            batch_weights = batch_weights.reshape(-1)
            # if l == n_loops - 1 and b == n_batch_seqs - 1 and s_idx == len(seq_inputs) - 1:
            #     print('log')
            # print("batch start loop: ", start_loop)
            yield batch_x, batch_y, batch_weights, batch_seq_len, start_loop

def iterate_batch_no_labels(inputs, batch_size, seq_length, shuffle_idx=None, augment_seq=False):
    """
    Generate a generator that return a batch of sequences of inputs and targets.

    This function randomly selects batches of multiple sequence. It then iterates
    through multiple sequence in parallel to generate a sequence of inputs and
    targets. It will append the input sequence with 0 and target with -1 when
    the lenght of each sequence is not equal.
    """
    # print(f"using batch_size{batch_size}, seq_length{seq_length}")
    n_inputs = len(inputs)

    if shuffle_idx is None:
        # No shuffle
        seq_idx = np.arange(n_inputs)
    else:
        # Shuffle subjects (get the shuffled indices from argument)
        seq_idx = shuffle_idx
    inputs = np.array(inputs)
    input_sample_shape = inputs[0].shape[1:]
    
    # Compute the number of maximum loops
    n_loops = int(math.ceil(len(seq_idx) / batch_size))  # todo ?

    # For each batch of subjects (size=batch_size)
    for l in range(n_loops):
        start_idx = l*batch_size
        end_idx = (l+1)*batch_size
        seq_inputs = np.asarray(inputs)[seq_idx[start_idx:end_idx]]

        if augment_seq:
            # Data augmentation: multiple sequences
            # Randomly skip some epochs at the beginning -> generate multiple sequence
            max_skips = 5
            for s_idx in range(len(seq_inputs)):
                n_skips = np.random.randint(max_skips)
                seq_inputs[s_idx] = seq_inputs[s_idx][n_skips:]

        # Determine the maximum number of batch sequences
        n_max_seq_inputs = -1
        for s_idx, s in enumerate(seq_inputs):
            if len(s) > n_max_seq_inputs:
                n_max_seq_inputs = len(s)

        n_batch_seqs = int(math.ceil(n_max_seq_inputs / seq_length))  # 一批个体（15个），最多有n_batch_seqs个seq， 每个seq的长度是20（seq_length）
        # For each batch sequence (size=seq_length)
        for b in range(n_batch_seqs):
            start_loop = True if b == 0 else False
            start_idx = b*seq_length
            end_idx = (b+1)*seq_length
            batch_inputs = np.zeros((batch_size, seq_length) + input_sample_shape, dtype=np.float32)  # batch_size=15, seq_length=20
            batch_weights = np.zeros((batch_size, seq_length), dtype=np.float32)
            batch_seq_len = np.zeros(batch_size, dtype=np.int64)
            # For each subject
            for s_idx, s in enumerate(seq_inputs):
                # (seq_len, sample_shape)
                each_seq_inputs = s[start_idx:end_idx]
                batch_inputs[s_idx, :len(each_seq_inputs)] = each_seq_inputs
                batch_weights[s_idx, :len(each_seq_inputs)] = 1
                batch_seq_len[s_idx] = len(each_seq_inputs)
            batch_x = batch_inputs.reshape((-1,) + input_sample_shape)
            batch_weights = batch_weights.reshape(-1)
            # if l == n_loops - 1 and b == n_batch_seqs - 1 and s_idx == len(seq_inputs) - 1:
            #     print('log')
            # print("batch start loop: ", start_loop)
            yield batch_x, batch_weights, batch_seq_len, start_loop
