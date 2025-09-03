# federated_align_curriculum_inbreast.py
import os
import torch
from torch import nn
import torch.optim as optim
import torch.distributions as tdist
import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader
from networks import Classifier, Discriminator
import params
from tensorboardX import SummaryWriter
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score
from dataset import get_loaders

warnings.filterwarnings("ignore")
EPS = 1e-12

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(federated_model, dataloader, train=False):
    federated_model.eval()
    val_running_loss = 0.0
    correct = 0
    probabilities = []
    predictions = []
    targets = []
    n_batches = 0

    with torch.no_grad():
        for n_batches, (inputs, labels, domain, idx) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            probs, logits = federated_model(inputs)
            preds = torch.argmax(probs, 1)
            loss = class_criterion(logits, labels)
            if torch.isnan(loss):  # skip bad batch
                continue
              # compute loss
            targets.append(labels.detach().cpu().numpy())
            probabilities.append(probs.detach().cpu().numpy())
            predictions.append(preds.detach().cpu().numpy())
            correct += preds.eq(labels.view(-1)).sum().item()
            val_running_loss += loss.item()
        # if no batch processed
        if n_batches == -1:
            return 0.0, 0.0, [], [], []
    # compute averages
    correct_total = correct / max(1, len(dataloader.dataset))
    if n_batches >= 0:
        val_running_loss = val_running_loss / (n_batches + 1)
    else:
        val_running_loss = 0.0

    if train:
        print('Train set local: Average loss: {:.4f}, Average acc: {:.4f}'.format(val_running_loss, correct_total))
    else:
        print('Test set local: Average loss: {:.4f}, Average acc: {:.4f}'.format(val_running_loss, correct_total))
    return val_running_loss, correct_total, targets, probabilities, predictions


def get_predictions(model, dataloader, n_train_val):
    model.eval()
    correct_predictions = np.zeros(n_train_val, dtype=np.int32)

    # If using a Subset, get the original dataset indices
    try:
        indices = dataloader.dataset.indices
    except AttributeError:
        indices = np.arange(len(dataloader.dataset))

    with torch.no_grad():
        for inputs, labels, domain, idx in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            probs, logits = model(inputs)
            correct_preds = torch.eq(labels, torch.argmax(probs, dim=1)).int().cpu().numpy()

            # Use Subset indices instead of batch idx
            batch_indices = idx.numpy() if isinstance(idx, torch.Tensor) else np.array(idx)
            correct_predictions[batch_indices] = correct_preds

    return correct_predictions


def get_curriculum_weights(preds_back, preds_recent):
    comparison = preds_back > preds_recent
    weights = comparison.astype(float) + 1.0
    return weights


# ----- single site -----
sites = [f"site{i}" for i in range(params.n_sites)]
n_sites = len(sites)

# model path for saving
PATH = './models/fed-align-cl/' + str(params.noise) + '/' + str(params.nsteps) + '-' + str(params.pace) + '/torch-seed-' + str(params.torch_seed)
print("Model PATH:", PATH)

# setup models & optimizers
global_model = Classifier().to(device)

local_models = {}
discriminators = {}
optimizers = {}
optimizerGs = {}
optimizerDs = {}

for i in range(n_sites):
    local_models[i] = Classifier().to(device)
    discriminators[i] = Discriminator().to(device)
    optimizers[i] = optim.Adam(local_models[i].parameters(), lr=params.learning_rate)

    try:
        optimizerGs[i] = optim.Adam(local_models[i].encoder.parameters(), lr=params.learning_rate)
    except Exception:
        optimizerGs[i] = optim.Adam(local_models[i].parameters(), lr=params.learning_rate)

    optimizerDs[i] = optim.Adam(discriminators[i].parameters(), lr=params.learning_rate)

# loss functions
class_criterion = nn.CrossEntropyLoss()


def advDloss(d1, d2):
    # d1, d2 expected in (0,1)
    d1c = torch.clamp(d1, EPS, 1.0 - EPS)
    d2c = torch.clamp(d2, EPS, 1.0 - EPS)
    res = -torch.log(d1c).mean() - torch.log(1.0 - d2c).mean()
    return res


def advGloss(d1, d2):
    d1c = torch.clamp(d1, EPS, 1.0 - EPS)
    d2c = torch.clamp(d2, EPS, 1.0 - EPS)
    res = -torch.log(d1c).mean() - torch.log(d2c).mean()
    # res is scalar; no need to .mean() again
    return res


# ==== DATASET LOADING DARI FOLDER SPLIT ====
from dataset import get_loaders

train_loaders, val_loaders, test_loaders = [], [], []
for site in range(params.n_sites):
    tr, vl, te = get_loaders(site, params.batch_size, params.data_transform, num_workers=params.num_workers)
    train_loaders.append(tr); val_loaders.append(vl); test_loaders.append(te)

# load pretrained weights if requested
if params.pretrained:
    print('loading pretrained weights')
    image_only_parameters = dict()
    image_only_parameters["model_path"] = "models/pretrained/sample_image_model.p"
    image_only_parameters["view"] = "L-CC"
    image_only_parameters["use_heatmaps"] = False

    for i in range(n_sites):
        try:
            local_models[i].encoder.load_state_from_shared_weights(
                state_dict=torch.load(image_only_parameters["model_path"])["model"],
                view=image_only_parameters["view"],
            )
        except Exception as e:
            print("Warning: failed loading local encoder pretrained:", e)

    try:
        global_model.encoder.load_state_from_shared_weights(
            state_dict=torch.load(image_only_parameters["model_path"])["model"],
            view=image_only_parameters["view"],
        )
    except Exception as e:
        print("Warning: failed loading global encoder pretrained:", e)

# define weights to combine local models (equal weights)
w = {i: 1.0 / n_sites for i in range(n_sites)}

# Summary writers
writer_train = SummaryWriter(os.path.join(PATH, 'train'))
writer_val = SummaryWriter(os.path.join(PATH, 'val'))

best_val_loss = np.inf
track_preds = dict()

# compute n_train_val for each loader
n_train_val = [len(tl.dataset) for tl in train_loaders]

print('Start optimization')
for epoch in range(params.n_epochs):    
    track_preds[epoch] = dict()

    # create iterators for each train loader (may be re-created later for curriculum)
    data_inters = [iter(tl) for tl in train_loaders]

    # compute predictions on training partitions for curriculum (if needed)
    for i in range(n_sites):
        dataset_i = train_loaders[i].dataset
        batch_sz_i = max(1, len(dataset_i) // params.nsteps)  # batch utk evaluasi kurikulum
        train_loader_eval = DataLoader(
            dataset_i,
            batch_size=batch_sz_i,
            shuffle=False,
            num_workers=params.num_workers
        )
        correct_preds_local  = get_predictions(local_models[i], train_loader_eval, len(dataset_i))
        correct_preds_global = get_predictions(global_model,     train_loader_eval, len(dataset_i))
        track_preds[epoch][i] = correct_preds_local
        local_models[i].train()

    # Curriculum sampling (only if we have previous epoch predictions)
    if epoch > params.n_epochs_adversarial:
        curriculum_weights = []
        for i in range(n_sites):
            prev = track_preds.get(epoch - 1, {}).get(i)
            curr = track_preds[epoch][i]
            # If prev absent, fallback to uniform
            if prev is None:
                weights = np.ones(len(curr))
            else:
                weights = get_curriculum_weights(prev, curr)
            curriculum_weights.append(weights)

        # create new WeightedRandomSampler-backed loaders
        new_train_loaders = []
        for i in range(n_sites):
            dataset_i = train_loaders[i].dataset
            wts = curriculum_weights[i]
            # jaga-jaga kalau panjang weights tidak sama (meski harusnya sama)
            if len(wts) != len(dataset_i):
                # fallback uniform
                import numpy as np
                wts = np.ones(len(dataset_i), dtype=float)

            sampler = WeightedRandomSampler(weights=wts, num_samples=len(dataset_i), replacement=True)
            batch_sz_i = max(1, len(dataset_i) // params.nsteps)
            new_loader = DataLoader(
                dataset_i,
                batch_size=batch_sz_i,
                shuffle=False,
                num_workers=params.num_workers,
                sampler=sampler
            )
            new_train_loaders.append(new_loader)

        train_loaders = new_train_loaders
        data_inters = [iter(tl) for tl in train_loaders]

    # prepare loss/metrics holders
    loss_all = {i: 0.0 for i in range(n_sites)}
    lossG_all = {i: 0.0 for i in range(n_sites)}
    lossD_all = {i: 0.0 for i in range(n_sites)}
    num_data = {i: EPS for i in range(n_sites)}
    num_dataG = {i: EPS for i in range(n_sites)}
    num_dataD = {i: EPS for i in range(n_sites)}

    for i in range(n_sites):
        local_models[i].train()
        discriminators[i].train()

    count = 0
    for t in range(params.nsteps):
        fs = []

        # optimize classifier (local training)
        for i in range(n_sites):
            optimizers[i].zero_grad()
            try:
                inputs, labels, domain, idx = next(data_inters[i])
            except StopIteration:
                # re-create iterator and continue
                data_inters[i] = iter(train_loaders[i])
                inputs, labels, domain, idx = next(data_inters[i])
            num_data[i] += labels.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            probs, logits = local_models[i](inputs)
            loss = class_criterion(logits, labels)
            loss_all[i] += float(loss.item()) * labels.size(0)
            loss.backward()
            optimizers[i].step()
            # append feature space
            try:
                fs.append(local_models[i].encoder(inputs))
            except Exception:
                # fallback: use whole model output as feature if no encoder attr
                fs.append(logits.detach())

        # optimize alignment (adversarial) - with single site this loop will skip updates (i != j)
        noises = []
        for i in range(n_sites):
            # if fs[i] is tensor, compute std; else fallback small constant
            try:
                std_val = 0.001 * float(torch.std(fs[i].detach().cpu()))
                nn_dist = tdist.Normal(torch.tensor([0.0]), std_val if std_val > 0 else 1e-6)
                noises.append(nn_dist.sample(fs[i].size()).squeeze().to(device))
            except Exception:
                noises.append(torch.zeros_like(fs[i]))

        for i in range(n_sites):
            for j in range(n_sites):
                if i == j:
                    continue
                optimizerDs[i].zero_grad()
                optimizerGs[i].zero_grad()
                optimizerGs[j].zero_grad()

                d1 = discriminators[i](fs[i].detach() + noises[i])
                d2 = discriminators[i](fs[j].detach() + noises[j])
                num_dataG[i] += d1.size(0)
                num_dataD[i] += d1.size(0)
                lossD = advDloss(d1, d2)
                lossG = advGloss(d1, d2)
                lossD_all[i] += float(lossD.item()) * d1.size(0)
                lossG_all[i] += float(lossG.item()) * d1.size(0)
                lossG_all[j] += float(lossG.item()) * d2.size(0)
                lossD = 0.1 * lossD

                if epoch >= params.n_epochs_adversarial:
                    lossG.backward(retain_graph=True)
                    optimizerGs[i].step()
                    optimizerGs[j].step()
                    lossD.backward(retain_graph=True)
                    optimizerDs[i].step()

                writer_train.add_histogram('Hist/hist_' + sites[i] + '2' + sites[j] + '_source', d1, epoch * params.nsteps + t)
                writer_train.add_histogram('Hist/hist_' + sites[i] + '2' + sites[j] + '_target', d2, epoch * params.nsteps + t)

        count += 1

        # communication - weight aggregation (here trivial since single site, but keep logic)
        if (count % params.pace == 0) or t == params.nsteps - 1:
            with torch.no_grad():
                for key in global_model.state_dict().keys():
                    if local_models[0].state_dict()[key].dtype == torch.int64:
                        global_model.state_dict()[key].data.copy_(local_models[0].state_dict()[key])
                    else:
                        temp = torch.zeros_like(global_model.state_dict()[key])
                        for s in range(n_sites):
                            if params.noise_type == 'G':
                                nn_dist = tdist.Normal(torch.tensor([0.0]), params.noise * torch.std(local_models[s].state_dict()[key].detach().cpu()))
                            else:
                                nn_dist = tdist.Laplace(torch.tensor([0.0]), params.noise * torch.std(local_models[s].state_dict()[key].detach().cpu()))
                            noise = nn_dist.sample(local_models[s].state_dict()[key].size()).squeeze(-1).to(device)
                            temp += w[s] * (local_models[s].state_dict()[key] + noise)
                        global_model.state_dict()[key].data.copy_(temp)
                        for s in range(n_sites):
                            local_models[s].state_dict()[key].data.copy_(global_model.state_dict()[key])

    # --- End step loop for epoch --- #
    print(f"Epoch Number {epoch + 1}")
    print("===========================")
    # Print per-site CE loss (generic)
    l_ce = [loss_all[i] / (num_data[i] if num_data[i] > 0 else 1.0) for i in range(n_sites)]
    l_g = [lossG_all[i] / (num_dataG[i] if num_dataG[i] > 0 else 1.0) for i in range(n_sites)]
    l_d = [lossD_all[i] / (num_dataD[i] if num_dataD[i] > 0 else 1.0) for i in range(n_sites)]
    print("CE loss per site:", ", ".join([f"{v:.7f}" for v in l_ce]))
    print("G loss per site:", ", ".join([f"{v:.7f}" for v in l_g]))
    print("D loss per site:", ", ".join([f"{v:.7f}" for v in l_d]))

    writer_train.add_scalars('CEloss', {f'l{i+1}': l_ce[i] for i in range(n_sites)}, epoch)
    writer_train.add_scalars('Gloss', {f'gl{i+1}': l_g[i] for i in range(n_sites)}, epoch)
    writer_train.add_scalars('Dloss', {f'dl{i+1}': l_d[i] for i in range(n_sites)}, epoch)

    # cleanup
    try:
        del fs, inputs
    except Exception:
        pass

    # --- Evaluation on validation set(s) ---
    all_targets = []
    all_probs = []
    aucs = []
    pr_aucs = []

    for i, vl in enumerate(val_loaders):
        print(f"=== {sites[i].upper()} ===")
        val_loss_i, acc_i, targets_i, probs_i, preds_i = test(global_model, vl, train=False)
        # flatten lists of arrays
        if len(targets_i) > 0:
            y_true = np.concatenate(targets_i).ravel()
            y_probs = np.concatenate(probs_i, axis=0)
            # if multi-class prob, take class 1 probability
            if y_probs.ndim == 2 and y_probs.shape[1] > 1:
                y_score = y_probs[:, 1]
            else:
                y_score = y_probs.ravel()
            try:
                auc = roc_auc_score(y_true, y_score)
            except Exception:
                auc = float('nan')
            try:
                pr = average_precision_score(y_true, y_score)
            except Exception:
                pr = float('nan')
        else:
            auc, pr, y_true, y_score = float('nan'), float('nan'), np.array([]), np.array([])

        print(f"Site {sites[i]} - AUC: {auc:.4f}, PR-AUC: {pr:.4f}, Acc: {acc_i:.4f}")
        writer_val.add_scalar(f'AUC/{sites[i]}', auc, epoch)
        writer_val.add_scalar(f'PR_AUC/{sites[i]}', pr, epoch)
        aucs.append(auc)
        pr_aucs.append(pr)

    # average metrics
    avg_auc = float(np.nanmean(aucs))
    avg_prauc = float(np.nanmean(pr_aucs))
    writer_val.add_scalar('AUC/avg', avg_auc, epoch)
    writer_val.add_scalar('PR_AUC/avg', avg_prauc, epoch)
    print(f"Average Val AUC: {avg_auc:.4f}, Average Val PR-AUC: {avg_prauc:.4f}")

    # Save model if validation loss improved (we approximate val loss by avg CE across sites)
    # compute average validation loss using previously computed l_ce
    average_train_loss = float(np.mean(l_ce))
    # we don't compute val_loss per site directly here; rely on AUC or use l_ce as proxy
    val_losses = []
    for i, vl in enumerate(val_loaders):
        val_loss_i, acc_i, targets_i, probs_i, preds_i = test(global_model, vl, train=False)
        val_losses.append(val_loss_i)
    average_val_loss = float(np.mean(val_losses)) if val_losses else np.inf


    writer_train.add_scalar('loss', average_train_loss, epoch)
    writer_val.add_scalar('loss', average_val_loss, epoch)

    if average_val_loss < best_val_loss:
        print('saving model')
        best_val_loss = average_val_loss
        if not os.path.exists(PATH):
            os.makedirs(PATH, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'global_model': global_model.state_dict(),
            'loss': best_val_loss,
        }, os.path.join(PATH, 'model.pt'))

print('Optimization finished!')
