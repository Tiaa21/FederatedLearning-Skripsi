# client.py
import torch, os, time, argparse
import params
from networks import Classifier, Discriminator
from dataset import get_loaders
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.distributions as tdist
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
import csv, time

EPS = 1e-10

# === timing init client model ===
init_start = time.time()
init_time = time.time() - init_start

# =====================================================
# Adversarial Loss
# =====================================================
def advDloss(d1, d2):
    d1c = torch.clamp(d1, EPS, 1.0 - EPS)
    d2c = torch.clamp(d2, EPS, 1.0 - EPS)
    res = -torch.log(d1c).mean() - torch.log(1.0 - d2c).mean()
    return res

def advGloss(d1, d2):
    d1c = torch.clamp(d1, EPS, 1.0 - EPS)
    d2c = torch.clamp(d2, EPS, 1.0 - EPS)
    res = -torch.log(d1c).mean() - torch.log(d2c).mean()
    return res

# =====================================================
# Test Function (match reference interface but fixed)
# =====================================================
def test(model, dataloader, train=False, criterion=None):
    """
    Return:
      val_running_loss (float), correct_ratio (float),
      targets (list of arrays per batch), probabilities (list), predictions (list)
    Matches reference signature/return shape but computes batch-average correctly.
    """
    model.eval()
    val_running_loss = 0.0
    correct = 0
    probabilities = []
    predictions = []
    targets = []
    batch_count = 0

    with torch.no_grad():
        for n_batches, (inputs, labels, domain, idx) in enumerate(dataloader):
            inputs = inputs.to(params.device)
            labels = labels.to(params.device)
            probs, logits = model(inputs)
            preds = torch.argmax(probs, 1)

            # compute loss if criterion given
            if criterion is not None:
                loss = criterion(logits, labels)
                val_running_loss += loss.item()
            else:
                # keep original style: still increment val_running_loss by 0 if not provided
                val_running_loss += 0.0

            targets.append(labels.detach().cpu().numpy())
            probabilities.append(probs.detach().cpu().numpy())
            predictions.append(preds.detach().cpu().numpy())

            correct += preds.eq(labels.view(-1)).sum().item()
            batch_count += 1

    # avoid division by zero; batch_count is number of batches
    if batch_count > 0:
        val_running_loss = val_running_loss / batch_count
    else:
        val_running_loss = 0.0

    # accuracy over dataset size
    dataset_len = len(dataloader.dataset)
    if dataset_len > 0:
        correct_ratio = correct / float(dataset_len)
    else:
        correct_ratio = 0.0

    if train:
        print('Train set local: Val loss: {:.4f}, Accuracy: {:.4f}'.format(val_running_loss, correct_ratio))
    else:
        print('Test set local: Val loss: {:.4f}, Accuracy: {:.4f}'.format(val_running_loss, correct_ratio))

    return val_running_loss, correct_ratio, targets, probabilities, predictions


# =====================================================
# Get Predictions (match ref behaviour but robust)
# =====================================================
def get_predictions(model, dataloader, n_train_val):
    """
    Returns array of 0/1 correct predictions per sample of dataset (length n_train_val),
    ordered according to dataloader.dataset.indices if exists (like Subset).
    """
    model.eval()
    correct_predictions = np.zeros(n_train_val, dtype=int)

    # train_indices: mapping from global-subset index to dataset index
    train_indices = getattr(dataloader.dataset, "indices", None)
    if train_indices is None:
        # dataset is not a Subset -> indices are 0..len-1
        train_indices = np.arange(len(dataloader.dataset))

    with torch.no_grad():
        for n_batches, (inputs, labels, domain, idx) in enumerate(dataloader):
            inputs = inputs.to(params.device)
            labels = labels.to(params.device)
            probs, logits = model(inputs)
            correct_preds = torch.eq(labels, torch.argmax(probs, dim=1)).int()
            # idx should be original dataset indices as returned by dataset.__getitem__
            # Convert idx to numpy array
            batch_idx = idx.detach().cpu().numpy() if isinstance(idx, torch.Tensor) else np.array(idx)
            correct_predictions[batch_idx] = correct_preds.detach().cpu().numpy()

    # reorder to match train_indices (like in reference)
    correct_predictions = correct_predictions[train_indices]
    return correct_predictions


# =====================================================
# Curriculum Weight
# =====================================================
def get_curriculum_weights(prev_preds, curr_preds):
    comp = prev_preds > curr_preds  # if was correct before but now wrong -> hard
    weights = comp.astype(float) + 1.0
    return weights


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_id", type=int, required=True)
    args = parser.parse_args()

    os.makedirs("agg", exist_ok=True)

    client_log = f"agg/client{args.site_id}_metrics.csv"
    if not os.path.exists(client_log):
        with open(client_log, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "Round",
                "Acc",
                "AUC",
                "PR-AUC",
                "Val Loss",
                "Training Time",
                "Upload (MB)",
                "Download (MB)",
            ])

    # TensorBoard logger
    writer = SummaryWriter(log_dir=f"runs/site{args.site_id}")

    # === Load data sesuai site ===
    train_loader, val_loader, test_loader = get_loaders(
        args.site_id,
        params.batch_size,
        params.data_transform,
        params.num_workers if hasattr(params, 'num_workers') else 0
    )

    model = Classifier()
    device = params.device
    model = model.to(device)

    # discriminator local
    disc = Discriminator().to(device)

    # optimizers
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    try:
        optimizerG = optim.Adam(model.encoder.parameters(), lr=params.learning_rate)
    except Exception:
        optimizerG = optim.Adam(model.parameters(), lr=params.learning_rate)
    optimizerD = optim.Adam(disc.parameters(), lr=params.learning_rate)

    class_criterion = torch.nn.CrossEntropyLoss()

    # optional pretrained encoder
    if getattr(params, 'pretrained', False):
        try:
            image_only_parameters = dict()
            image_only_parameters["model_path"] = "models/pretrained/sample_image_model.p"
            image_only_parameters["view"] = "L-CC"
            model.encoder.load_state_from_shared_weights(
                state_dict=torch.load(image_only_parameters["model_path"])["model"],
                view=image_only_parameters["view"],
            )
        except Exception as e:
            print("Warning: failed loading local encoder pretrained:", e)

    round_num = 0
    train_eval_loader, _, _ = get_loaders(
        args.site_id,
        params.batch_size,
        params.data_transform,
        params.num_workers if hasattr(params, 'num_workers') else 0
    )

    prev_correct = None

    while True:
        train_start = time.time()
        # === Accumulators ===
        loss_ce_total, loss_g_total, loss_d_total = 0.0, 0.0, 0.0
        n_ce, n_g, n_d = 0, 0, 0

        curriculum_enabled = getattr(params, 'use_curriculum', True)

        # === Curriculum correctness before training ===
        dataset_full = train_loader.dataset
        n_samples = len(dataset_full)

        batch_sz_eval = max(1, n_samples // getattr(params, 'nsteps', 20))
        train_eval_loader = DataLoader(
            dataset_full,
            batch_size=batch_sz_eval,
            shuffle=False,
            num_workers=getattr(params, 'num_workers', 0)
        )

        curr_correct = np.zeros(n_samples, dtype=np.int32)
        model.eval()
        with torch.no_grad():
            try:
                for inputs, labels, domain, idx in train_eval_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    probs, logits = model(inputs)
                    preds = torch.argmax(probs, dim=1).cpu().numpy()
                    batch_idx = idx.numpy() if isinstance(idx, torch.Tensor) else np.array(idx)
                    curr_correct[batch_idx] = (preds == labels.cpu().numpy()).astype(np.int32)
            except Exception:
                curr_correct = None

        if curr_correct is not None and prev_correct is not None and curriculum_enabled:
            weights = get_curriculum_weights(prev_correct, curr_correct)
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            train_loader = DataLoader(
                dataset_full,
                batch_size=params.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=getattr(params, 'num_workers', 0)
            )

        prev_correct = curr_correct

        # === LOCAL TRAINING ===
        model.train()
        data_iter = iter(train_loader)
        for t in range(getattr(params, 'nsteps', 20)):
            try:
                x, y, *rest = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y, *rest = next(data_iter)

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            logits = out[1] if isinstance(out, tuple) else out
            loss = class_criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_ce_total += float(loss.item()) * y.size(0)
            n_ce += y.size(0)
        local_train_time = time.time() - train_start

        # === Adversarial Alignment ===
        global_path = f"agg/global_round{max(0, round_num-1)}.pth"
        if os.path.exists(global_path):
            try:
                global_state = torch.load(global_path, map_location=device)
                global_model = Classifier().to(device)
                global_model.load_state_dict(global_state)
                global_model.eval()
            except Exception as e:
                print(f"[CLIENT {args.site_id}] Failed to load global model for adv: {e}")
                global_model = None
        else:
            global_model = None

        print(f"[CLIENT {args.site_id}] Check ADV: round={round_num}, global_model={global_model is not None}, n_epochs_adv={getattr(params, 'n_epochs_adversarial', 0)}")

        if global_model is not None and round_num >= getattr(params, 'n_epochs_adversarial', 0):
            model.train()
            disc.train()
            try:
                inputs, labels, domain, idx = next(iter(train_eval_loader))
            except Exception:
                inputs, labels = next(iter(train_loader))
            inputs = inputs.to(device)

            try:
                fs_local = model.encoder(inputs)
            except Exception:
                probs, logits = model(inputs)
                fs_local = logits

            with torch.no_grad():
                try:
                    fs_global = global_model.encoder(inputs)
                except Exception:
                    probs_g, logits_g = global_model(inputs)
                    fs_global = logits_g

            try:
                std_val = 0.001 * float(torch.std(fs_local.detach().cpu()))
                nn_dist = tdist.Normal(torch.tensor([0.0]), std_val if std_val > 0 else 1e-6)
                noise_local = nn_dist.sample(fs_local.size()).squeeze().to(device)
            except Exception:
                noise_local = torch.zeros_like(fs_local)

            try:
                std_valg = 0.001 * float(torch.std(fs_global.detach().cpu()))
                nn_distg = tdist.Normal(torch.tensor([0.0]), std_valg if std_valg > 0 else 1e-6)
                noise_global = nn_distg.sample(fs_global.size()).squeeze().to(device)
            except Exception:
                noise_global = torch.zeros_like(fs_global)

            optimizerD.zero_grad()
            d1 = disc(fs_local.detach() + noise_local)
            d2 = disc(fs_global.detach() + noise_global)
            lossD = advDloss(d1, d2)
            loss_d_total += float(lossD.item()) * d1.size(0)
            n_d += d1.size(0)
            lossD.backward()
            optimizerD.step()

            optimizerG.zero_grad()
            d1_forG = disc(fs_local + noise_local)
            d2_forG = disc(fs_global.detach() + noise_global)
            lossG = advGloss(d1_forG, d2_forG)
            loss_g_total += float(lossG.item()) * d1.size(0)
            n_g += d1.size(0)
            lossG.backward()
            optimizerG.step()

        # === Compute averages ===
        avg_ce = loss_ce_total / max(1, n_ce)
        avg_g  = loss_g_total / max(1, n_g)
        avg_d  = loss_d_total / max(1, n_d)

        # === Local evaluation ===
        val_loss, acc_ratio, targets, probs_list, preds_list = test(model, val_loader, train=False, criterion=class_criterion)

        # Compute ROC/PR/CM here from concatenated probs/targets to avoid double printing in test()
        try:
            all_targets = np.concatenate(targets)
            all_probs = np.concatenate(probs_list)
            roc_auc = roc_auc_score(all_targets, all_probs[:, 1])
            pr_auc = average_precision_score(all_targets, all_probs[:, 1])
            cm = confusion_matrix(all_targets, np.concatenate(preds_list))
        except Exception:
            roc_auc, pr_auc, cm = float('nan'), float('nan'), None

        # simpan metrics
        metrics_path = f"agg/site{args.site_id}_round{round_num}_metrics.npy"
        np.save(metrics_path, {
            "acc": acc_ratio,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "val_loss": val_loss
        })

        # === TensorBoard logging ===
        writer.add_scalar("Val/Loss", val_loss, round_num)
        writer.add_scalar("Loss/CE", avg_ce, round_num)
        writer.add_scalar("Loss/G", avg_g, round_num)
        writer.add_scalar("Loss/D", avg_d, round_num)
        writer.add_scalar("Val/Acc", acc_ratio, round_num)
        writer.add_scalar("Val/ROC_AUC", roc_auc, round_num)
        writer.add_scalar("Val/PR_AUC", pr_auc, round_num)

        # === Print summary (single place) ===
        print(
            f"[CLIENT {args.site_id}] Round {round_num} LOCAL – "
            f"CE: {avg_ce:.4f}, G: {avg_g:.4f}, D: {avg_d:.4f}, "
            f"ValLoss: {val_loss:.4f}, Acc: {acc_ratio:.4f}, AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}"
        )

        # save model weights
        local_path = f"agg/site{args.site_id}_round{round_num}.pth"

        comm_start = time.time()

        torch.save(model.state_dict(), local_path)
        print(f"[CLIENT {args.site_id}] Uploaded round {round_num}")

        # hitung size upload
        upload_size = os.path.getsize(local_path) / (1024.0 * 1024.0)  # MB

        # === Tunggu global sync ===
        next_global = f"agg/global_round{round_num}.pth"
        while not os.path.exists(next_global):
            time.sleep(2)

        # hitung size download
        download_size = os.path.getsize(next_global) / (1024.0 * 1024.0)  # MB
        comm_time = time.time() - comm_start

        with open(client_log, "a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                round_num,
                acc_ratio,       
                roc_auc,         
                pr_auc,
                val_loss,
                local_train_time,
                upload_size,
                download_size,
            ])

        print("================== ⏱ TIME MODEL ⏱ ==================")
        print(f"[CLIENT {args.site_id}] Upload: {upload_size:.4f} MB, Download: {download_size:.4f} MB, Training Time: {local_train_time:.4f}s")

        while True:
            try:
                state_dict = torch.load(next_global, map_location=device, weights_only=False)
                model.load_state_dict(state_dict)
                break
            except Exception as e:
                print(f"[CLIENT {args.site_id}] Global file invalid, retrying... ({e})")
                time.sleep(2)


        print(f"[CLIENT {args.site_id}] Synced global round {round_num}")

        round_num += 1