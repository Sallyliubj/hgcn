from argparse import Namespace

args = Namespace(
    task = 'lp',
    dataset = 'inference',
    model = 'HGCN',
    lr = 0.01,
    dim = 16,
    num_layers = 2,
    act = 'relu',
    bias = 1,
    dropout = 0.4,
    weight_decay = 0.0001,
    manifold = 'PoincareBall',
    log_freq = 5,
    save = 1,
    seed = 1234,
    double_precision = '0',
    cuda = -1,
    patience = 100,
    save_dir =  None,
    val_prop = 0.05,
    test_prop = 0.1,
    split_seed = 1234,
    normalize_adj = 1,
    normalize_feats = 1,
    use_feats = 1,
    lr_reduce_freq = None,
    epochs = 2000,
    min_epochs = 100,
    gamma = 0.5,
    optimizer = 'Adam',
    c = 1.0,
    r = 2.,
    t = 1.,
    n_classes = 3,
    use_att = 0,
    local_agg = 0,
    n_heads = 4,
    alpha = 0.2,
    pos_weight = 0,
    grad_clip = None,
    eval_freq = 1,
)