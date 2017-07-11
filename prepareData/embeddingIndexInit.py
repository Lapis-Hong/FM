def embedding_index_init():
    """
    # chose the features after one-hot that to be the embedding vectors
    :return: embedding index
    """
    a1 = range(23, 33)
    a2 = range(33, 42)
    a3 = range(42, 62)
    a4 = range(62, 79)
    a5 = range(79, 99)
    a6 = range(99, 118)
    a7 = range(130, 148)
    a7_1 = range(254, 262)
    a7_2 = range(349, 351)
    a7_3 = range(352, 358)
    a8 = range(158, 173)
    a8_1 = range(641, 646)
    a9 = range(173, 187)
    a9_1 = range(646, 650)
    a10 = range(187, 207)
    a11 = range(213, 233)
    a12 = range(233, 253)
    a13 = range(268, 277)
    a14 = range(363, 383)
    a15 = range(383, 403)
    a16 = range(405, 419)
    a17 = range(419, 432)
    a18 = range(439, 456)
    a19 = range(456, 472)
    a20 = range(472, 486)
    a21 = range(489, 498)
    a22 = range(498, 510)
    a23 = range(514, 525)
    a24 = range(525, 535)
    a25 = range(553, 565)
    a26 = range(565, 581)
    a27 = range(586, 600)
    a28 = range(600, 612)
    a29 = range(612, 621)
    a30 = range(650, 663)
    a31 = range(691, 701)
    a32 = range(701, 710)
    a33 = range(710, 720)
    a34 = range(720, 734)
    a35 = range(734, 744)
    a36 = range(744, 753)
    a37 = range(753, 765)
    a38 = range(772, 780)
    a39 = range(780, 788)
    embedding_index = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a7_1 + a7_2 + a7_3 + a8 + a8_1 + a9 + a9_1 \
                      + a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23 \
                      + a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31 + a32 + a33 + a34 + a35 + a36 + a37 + a38 + a39
    return embedding_index


def not_embedding_index():
    index_all = range(788 + 1)
    embedding_index = embedding_index_init()
    return list(set(index_all) - set(embedding_index))
