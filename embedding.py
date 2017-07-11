from cPickle import dump, load

def load_latent(infile):
    """
    :param infile: latent vector path
    :return: list format latent vector, 
    like [[-0.145157, -0.101264, 0.0266148, -0.123098, -0.10167, 0.10267, 0.0328743, -0.20144], [-0.105264, -0.
    """
    with open(infile) as model:
        f = model.read()
        # latentVec = [[str(w) for w in line] for line in latentVec]
    return eval(f)


# general save data function
def save_data(data, file_path):
    with open(file_path, 'w') as f:
        f.write(str(data))


def latent_embedding(infile, latent):
    """
    take the fm latent vectors as the embedding features, generate new data
    :param infile: prd data path with libfm format
    :param latent: latent vector
    :param index_list: embedding feature index list
    :return: new data with embedding features
    """
    new_data = []
    with open(infile) as f:
        for line in f:
            embedding = []
            temp = line.strip('\n').split(' ')
            embedding.append(temp.pop(0))  # add order_id to the first col, string
            embedding_index = [item.split(':')[0] for item in temp if float(item.split(':')[1])==1]
            for w in embedding_index:
                embedding.extend(latent[int(w)])
            new_data.append(embedding)
    return new_data





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


