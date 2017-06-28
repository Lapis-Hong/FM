def load_latent(file_path):
    '''
    :param file_path: latent vector path
    :return: list format latent vector, 
    like [[-0.145157, -0.101264, 0.0266148, -0.123098, -0.10167, 0.10267, 0.0328743, -0.20144], [-0.105264, -0.
    '''
    with open(file_path) as model:
        f = model.read()
        # latentVec = [[str(w) for w in line] for line in latentVec]
    return eval(f)


def latent_embedding(file_path, latent, index_list):
    '''
    take the fm latent vectors as the embedding features, generate new data
    :param file_path: train data path
    :param latent: latent vector
    :param index_list: embedding feature index list
    :return: new data with embedding features
    '''
    new_data = []
    with open(file_path) as train_data:
        for line in train_data:
            embedding_feature = []
            temp = line.strip('\n').split(' ')
            embedding_feature.append(temp.pop(0))  # add target to the first col
            feature_index = [int(w.split(':')[0]) for w in temp if int(
                w.split(':')[0]) in index_list]  # save the embedding feature from training data(with value 1)
            for embedding_ind in index_list:
                if embedding_ind in feature_index:
                    embedding_feature.extend(
                        latent[embedding_ind])  # index from zero, and float format data for latentVec
                else:
                    embedding_feature.extend([0.0] * len(latent[0]))
            new_data.append(embedding_feature)
    return new_data


# general save data function
def save_data(data, file_path):
    with open(file_path, 'w') as f:
        f.write(str(data))


def gen_cmd_line(train_path, test_path):
    cmdline = []
    # essential
    cmdline.extend(["./libFM", "-task", "c", "-train", train_path,
                    "-test", test_path, "-dim", "1,1,8"])
    # parameter
    cmdline.extend(["-iter", "234", "-method", "sgd", "-learn_rate",
                    "0.01", "-regular", "'0,0,0.01'", "-init_stdev", "0.1"])
    # save model and predict
    cmdline.extend(["-save_model", "model", "-out", "output"])
    return cmdline
