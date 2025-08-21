def accuracy(y_true, y_pred):
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return correct / len(y_true)
