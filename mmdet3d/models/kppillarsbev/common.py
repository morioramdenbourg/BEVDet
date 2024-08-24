def convert_to_feature_order(tensor):
    batch_dim = 0
    length_dim = 1
    feature_dim = 2
    return tensor.permute(batch_dim, feature_dim, length_dim)

def convert_to_length_order(tensor):
    batch_dim = 0
    feature_dim = 1
    length_dim = 2
    return tensor.permute(batch_dim, length_dim, feature_dim)


