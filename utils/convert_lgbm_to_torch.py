import pickle

import hummingbird
from hummingbird.ml import constants


# DATA_HOME_DIR = "/Users/pprakash/PycharmProjects/prob-ent-resolution/data/S2AND"
DATA_HOME_DIR = "/work/pi_mccallum_umass_edu/pragyaprakas_umass_edu/prob-ent-resolution/data"

def convert_pretrained_model():
    with open(f"{DATA_HOME_DIR}/production_model.pickle", "rb") as _pkl_file:
        chckpt = pickle.load(_pkl_file)
        clusterer = chckpt['clusterer']

    # Get Classifier and convert to torch model
    lgbm = clusterer.classifier
    print(lgbm)
    torch_model = hummingbird.ml.convert(clusterer.classifier, "torch", None,
                                             extra_config=
                                             {constants.FINE_TUNE: True,
                                              constants.FINE_TUNE_DROPOUT_PROB: 0.1})
    return torch_model.model


