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
#DATA_HOME_DIR = "../data"
DATA_HOME_DIR = "/work/pi_mccallum_umass_edu/pragyaprakas_umass_edu/prob-ent-resolution/data"

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


def load_pretrained_model():
    with open(f"{DATA_HOME_DIR}/production_model.pickle", "rb") as _pkl_file:
        chckpt = pickle.load(_pkl_file)
        clusterer = chckpt['clusterer']

    # Get Classifier to convert to torch model
    lgbm = clusterer.classifier
    print(lgbm)

    # Predict using this chckpt
    # y_proba = lgbm.predict_proba(Xtrain)[:, 1]
    # print(y_proba)

    torch_model = hummingbird.ml.convert(clusterer.classifier, "torch", None,
                                             extra_config=
                                             {constants.FINE_TUNE: True,
                                              constants.FINE_TUNE_DROPOUT_PROB: 0.1})
    return lgbm, torch_model.model

def predict_proba(model, input):
    return model(input)[1][:, 1]
def evaluate(model, input, output):
    return (sum(model(input)[0] == output) / len(input)).item()

def finetune_torch_model(lgbm, torch_lgbm, Xtrain, Xtest, Ytrain, Ytest):
    # Print out sizes of each layer
    print("LGBM converted to torch model with following structure")
    for name, param in torch_lgbm.named_parameters():
        print(name, param.size())

        # Do fine tuning
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(torch_lgbm.parameters(), lr=1e-3, weight_decay=5e-4)
        X_tensor = torch.tensor(Xtrain)
        y_tensor = torch.from_numpy(Ytrain).float()
        X_test_tensor = torch.tensor(Xtest)
        y_test_tensor = torch.from_numpy(Ytest).int()


        print("Original loss: ",
              loss_fn(torch.from_numpy(lgbm.predict_proba(Xtrain)[:, 1]).float(), y_tensor).item())
        with torch.no_grad():
            torch_lgbm.eval()
            print("Fine-tuning starts from training loss: ", evaluate(torch_lgbm, X_tensor.to(device), y_tensor.to(device)) * 100,
                  "%")
        torch_lgbm.train()

        batch_size = 20000
        for i in range(20):  # epoch
            for j in range(0, len(X_tensor), batch_size):  # batch
                X_batch = X_tensor[j:j + batch_size].to(device)
                y_batch = y_tensor[j:j + batch_size].to(device)

                optimizer.zero_grad()
                y_ = predict_proba(torch_lgbm, X_batch)
                assert y_.requires_grad

                loss = loss_fn(y_, y_batch)
                loss.backward()
                optimizer.step()

                # Print batch loss
                with torch.no_grad():
                    torch_lgbm.eval()
                    print("\tBatch", f"[{j}:{j + batch_size}]", ":",
                          loss_fn(predict_proba(torch_lgbm, X_batch), y_batch).item())
                torch_lgbm.train()

            # Print epoch validation accuracy
            with torch.no_grad():
                torch_lgbm.eval()
                print("Epoch", i + 1, ":", "Test accuracy:",
                      evaluate(torch_lgbm, X_test_tensor.to(device), y_test_tensor.to(device)) * 100, "%")
            torch_lgbm.train()


        with torch.no_grad():
            torch_lgbm.eval()
            print("Fine-tuning completed with training loss: ", evaluate(torch_lgbm, X_tensor.to(device), y_tensor.to(device)) * 100, "%")
            # confmat = ConfusionMatrix(num_classes=2)
            # print("Confusion Matrix of finetuned torch model")
            # print(confmat(torch_lgbm(Xtest)[1][:, 1], y_test_tensor))

        with torch.no_grad():
            torch_lgbm.eval()
            print("----------------")
            print("Final Test accuracy:", evaluate(torch_lgbm, X_test_tensor.to(device), y_test_tensor.to(device)) * 100, "%")
            print("----------------")


if __name__=='__main__':
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device={device}")

    # Load dataset for S2AND
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_featurize_dataset()
    print("Data Featurized and Ready")

    # Load and convert pretrained LGBM to Torch
    lgbm, torch_lgbm = load_pretrained_model()
    print("Model loaded and converted to Torch")
    # Finetune this converted model and compare losses
    finetune_torch_model(lgbm, torch_lgbm, X_train, X_test, y_train, y_test)

