import pickle
import hummingbird.ml
from hummingbird.ml import constants
from os.path import join
from s2and.data import ANDData
from s2and.model import PairwiseModeler
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.eval import pairwise_eval
import torch
from torchmetrics import ConfusionMatrix



# TO-DO: Change this to some local directory
DATA_HOME_DIR = "../data"

def load_and_featurize_dataset():
    dataset_name = "arnetminer"
    parent_dir = f"{DATA_HOME_DIR}/{dataset_name}"
    dataset = ANDData(
        signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
        papers=join(parent_dir, f"{dataset_name}_papers.json"),
        mode="train",
        specter_embeddings=join(parent_dir, f"{dataset_name}_specter.pickle"),
        clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
        block_type="s2",
        train_pairs_size=10000,
        val_pairs_size=0,
        test_pairs_size=1000,
        name=dataset_name,
        n_jobs=4,
    )

    # Load the featurizer, which calculates pairwise similarity scores
    featurization_info = FeaturizationInfo()
    # the cache will make it faster to train multiple times - it stores the features on disk for you
    train, val, test = featurize(dataset, featurization_info, n_jobs=4, use_cache=True)
    X_train, y_train, _ = train
    X_val, y_val, _ = val
    X_test, y_test, _ = test

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_pretrained_model(Xtrain):
    dataset_name = "arnetminer"
    parent_dir = f"{DATA_HOME_DIR}/{dataset_name}"
    dataset = ANDData(
        signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
        papers=join(parent_dir, f"{dataset_name}_papers.json"),
        mode="inference",
        specter_embeddings=join(parent_dir, f"{dataset_name}_specter.pickle"),
        block_type="s2",
        name=dataset_name,
    )

    with open(f"{DATA_HOME_DIR}/production_model.pickle", "rb") as _pkl_file:
        chckpt = pickle.load(_pkl_file)
        clusterer = chckpt['clusterer']

    # Get Classifier to convert to torch model
    lgbm = clusterer.classifier
    print(lgbm)

    # Predict using this chckpt
    # y_proba = lgbm.predict_proba(Xtrain)[:, 1]
    # print(y_proba)

    #lgbm.fit()
    torch_model = hummingbird.ml.convert(clusterer.classifier, "torch", None,
                                             extra_config=
                                             {constants.FINE_TUNE: True,
                                              constants.FINE_TUNE_DROPOUT_PROB: 0.1})
    return lgbm, torch_model.model

def finetune_torch_model(lgbm, torch_model, Xtrain, Xtest, Ytrain, Ytest):
    # Print out sizes of each layer
    print("LGBM converted to torch model with following structure")
    for name, param in torch_model.named_parameters():
        print(name, param.size())

        # Do fine tuning
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(torch_model.parameters(), lr=1e-3, weight_decay=5e-4)
        y_tensor = torch.from_numpy(Ytrain).float()
        y_test_tensor = torch.from_numpy(Ytest).int()

        print("Original loss: ",
              loss_fn(torch.from_numpy(lgbm.predict_proba(Xtrain)[:, 1]).float(), y_tensor).item())
        with torch.no_grad():
            torch_model.eval()
            print("Fine-tuning starts from loss: ", loss_fn(torch_model(Xtrain)[1][:, 1], y_tensor).item())
        torch_model.train()

        for i in range(200):
            optimizer.zero_grad()
            y_ = torch_model(Xtrain)[1][:, 1]
            loss = loss_fn(y_, y_tensor)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                with torch.no_grad():
                    torch_model.eval()
                    print("Iteration ", i, ": ", loss_fn(torch_model(Xtrain)[1][:, 1], y_tensor).item())
                torch_model.train()

        with torch.no_grad():
            torch_model.eval()
            print("Fine-tuning done with loss: ", loss_fn(torch_model(Xtrain)[1][:, 1], y_tensor).item())
            confmat = ConfusionMatrix(num_classes=2)
            print("Confusion Matrix of finetuned torch model")
            print(confmat(torch_model(Xtest)[1][:, 1], y_test_tensor))


if __name__=='__main__':
    # Load dataset for S2AND
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_featurize_dataset()
    print("Data Featurized and Ready")

    # Load and convert pretrained LGBM to Torch
    lgbm, torch_lgbm = load_pretrained_model(X_train)
    print("Model loaded and converted to Torch")
    # Finetune this converted model and compare losses
    finetune_torch_model(lgbm, torch_lgbm, X_train, X_test, y_train, y_test)

