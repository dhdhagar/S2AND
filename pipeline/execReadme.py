from os.path import join
from s2and.data import ANDData

dataset_name = "arnetminer"
parent_dir = f"../data/{dataset_name}"

dataset = ANDData(
    signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
    papers=join(parent_dir, f"{dataset_name}_papers.json"),
    mode="train",
    specter_embeddings=join(parent_dir, f"{dataset_name}_specter.pickle"),
    clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
    block_type="s2",
    train_pairs_size=100,
    val_pairs_size=100,
    test_pairs_size=100,
    name=dataset_name,
    n_jobs=4,
)
print("loaded the data")

# Training Featurizer model
from s2and.model import PairwiseModeler
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.eval import pairwise_eval, cluster_eval

featurization_info = FeaturizationInfo()
# the cache will make it faster to train multiple times - it stores the features on disk for you
train, val, test = featurize(dataset, featurization_info, n_jobs=4, use_cache=True)
X_train, y_train, _ = train
X_val, y_val, _ = val
X_test, y_test, _ = test
print("Done loading and featurizing")

# calibration fits isotonic regression after the binary classifier is fit
# monotone constraints help the LightGBM classifier behave sensibly
pairwise_model = PairwiseModeler(
    n_iter=25, monotone_constraints=featurization_info.lightgbm_monotone_constraints
)
# this does hyperparameter selection, which is why we need to pass in the validation set.
pairwise_model.fit(X_train, y_train, X_val, y_val)
print("Fitted the Pairwise model")

# this will also dump a lot of useful plots (ROC, PR, SHAP) to the figs_path
pairwise_metrics = pairwise_eval(X_test, y_test, pairwise_model.classifier, figs_path='figs/', title='example')
print(pairwise_metrics)

# # Clustering
# from s2and.model import Clusterer, FastCluster
# from hyperopt import hp
#
# clusterer = Clusterer(
#     featurization_info,
#     pairwise_model,
#     cluster_model=FastCluster(linkage="average"),
#     search_space={"eps": hp.uniform("eps", 0, 1)},
#     n_iter=25,
#     n_jobs=8,
# )
# clusterer.fit(dataset)
#
# # the metrics_per_signature are there so we can break out the facets if needed
# metrics, metrics_per_signature = cluster_eval(dataset, clusterer)
# print(metrics)


