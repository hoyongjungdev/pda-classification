def f1(cf):
    tp = cf[1, 1]
    fp = cf[0, 1]
    fn = cf[1, 0]

    if tp == 0:
        return 0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return 2 * recall * precision / (recall + precision)
