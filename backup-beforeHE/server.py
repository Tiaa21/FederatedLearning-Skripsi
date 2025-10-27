# server.py
import torch, os, time, argparse
from networks import Classifier
from dataset import get_loaders
import params
import torch.distributions as tdist
import numpy as np
import shutil, errno
from sklearn.metrics import roc_auc_score, average_precision_score
import math
from torch.utils.tensorboard import SummaryWriter
import csv, time

best_val_loss = float("inf")
patience = getattr(params, "patience", 10)  # bisa set di params.py
bad_epochs = 0

FRESH_DIRS = ["agg", "runs"]

for d in FRESH_DIRS:
# recreate agg dir (like original)
    if os.path.exists("agg"):
        try:
            shutil.rmtree(d)
            print(f"Folder {d} berhasil dihapus!")
        except OSError as e:
            print(f"⚠️ Gagal hapus agg: {e}, coba hapus manual.")
    os.makedirs("agg", exist_ok=True)
    print(f"Folder {d} berhasil dibuat ulang")

# === CREATE CSV ===
logfile = "agg/global_metrics.csv"
# create logfile with extended header if not exists
if not os.path.exists(logfile):
    with open(logfile, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "Round",
            "Avg Acc",
            "Avg AUC",
            "Avg PR-AUC",
            "Avg Val Loss",
            "Agg Time",
            "Init Time",
            "Consumption (MB)",
            "Epoch Time",
            "Cumulative Time"
        ])

# === timing init global model ===
init_start = time.time()
global_model = Classifier().to(params.device)
init_time = time.time() - init_start

# === TensorBoard writer untuk global metrics ===
writer = SummaryWriter(log_dir="runs/server")

def safe_metrics(y_true, y_score):
    try:
        if len(y_true) == 0:
            return float('nan'), float('nan')
        if len(np.unique(y_true)) < 2:
            return float('nan'), float('nan')
        if np.any(np.isnan(y_score)) or np.any(np.isnan(y_true)):
            return float('nan'), float('nan')
        auc = roc_auc_score(y_true, y_score)
        pr = average_precision_score(y_true, y_score)
        return auc, pr
    except Exception:
        return float('nan'), float('nan')

def aggregate_with_noise(weights_list, noise=0.0, noise_type='G', w=None, device=torch.device('cpu')):
    # weights_list: list of state_dicts
    # w: dict of per-site weights (default equal)
    if w is None:
        n = len(weights_list)
        w = {i: 1.0 / n for i in range(len(weights_list))}
    avg = {}
    # convert to local tensors on device
    for key in weights_list[0].keys():
        # if integer tensor (e.g., indices) simply copy from first
        if weights_list[0][key].dtype == torch.int64:
            avg[key] = weights_list[0][key].clone()
            continue
        temp = torch.zeros_like(weights_list[0][key], device=device, dtype=weights_list[0][key].dtype)
        for s, st in enumerate(weights_list):
            param = st[key].to(device)
            if noise and noise > 0:
                # compute std robustly
                param_std = float(torch.std(param.detach().cpu()))
                std = noise * param_std if param_std > 0 else noise * 1e-6
                if noise_type == 'G':
                    nn_dist = tdist.Normal(torch.tensor([0.0]), torch.tensor([std]))
                else:
                    nn_dist = tdist.Laplace(torch.tensor([0.0]), torch.tensor([std]))
                sampled = nn_dist.sample(param.size()).squeeze(-1).to(device)
                param = param + sampled
            temp += w[s] * param
        avg[key] = temp.clone().cpu()
    return avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--eval_site", type=int, default=0, help="site ID untuk evaluasi global (ignored in multi-site)")  
    args = parser.parse_args()

    os.makedirs("agg", exist_ok=True)
    global_model = Classifier().to(params.device)

    cumulative_start = time.time()

    # Use for-loop over rounds (0..n_epochs-1)
    for round_num in range(params.n_epochs):
        epoch_start = time.time()

        weights_list = []
        # wait & load client uploads for this round
        for i in range(args.num_clients):
            local_path = f"agg/site{i}_round{round_num}.pth"
            while not os.path.exists(local_path):
                time.sleep(2)
            weights = torch.load(local_path, map_location='cpu')  # load to cpu
            weights_list.append(weights)

        # === Aggregasi with noise ===
        w = {i: 1.0 / args.num_clients for i in range(args.num_clients)}

        agg_start = time.time()

        avg_state = aggregate_with_noise(
            weights_list,
            noise=getattr(params, 'noise', 0.0),
            noise_type=getattr(params, 'noise_type', 'G'),
            w=w,
            device=params.device
        )

        agg_time = time.time() - agg_start

        global_model.load_state_dict(avg_state)

        # debug NaNs
        for name, param in global_model.state_dict().items():
            if torch.isnan(param).any():
                print(f"[DEBUG] NaN detected in layer {name}")

        # === Save global model so clients can sync ===
        global_path = f"agg/global_round{round_num}.pth"
        tmp_path = global_path + ".tmp"
        torch.save(global_model.state_dict(), tmp_path)
        os.replace(tmp_path, global_path)  # atomic rename
        print(f"[SERVER] Round {round_num} aggregated → {global_path}")
        

        # === Tunggu semua client ambil model dan upload next round (synchronization) ===
        if round_num < params.n_epochs - 1:
            next_round_ready = False
            while not next_round_ready:
                ready_count = 0
                for i in range(args.num_clients):
                    next_path = f"agg/site{i}_round{round_num+1}.pth"
                    if os.path.exists(next_path):
                        ready_count += 1
                if ready_count == args.num_clients:
                    next_round_ready = True
                else:
                    time.sleep(2)

        # === compute consumption MB as sum of sizes of client files (approx comm cost) ===
        consumption_bytes = 0
        for p in local_path:
            try:
                consumption_bytes += os.path.getsize(p)
            except Exception:
                pass
        consumption_MB = consumption_bytes / (1024.0 * 1024.0)

        # === Evaluasi global (server kumpulin metrics dari client) ===
        all_acc, all_auc, all_pr, all_val_loss = [], [], [], []
        for site_id in range(args.num_clients):
            metrics_path = f"agg/site{site_id}_round{round_num}_metrics.npy"
            while not os.path.exists(metrics_path):
                time.sleep(2)
            metrics = np.load(metrics_path, allow_pickle=True).item()

            # Safely get metrics with defaults (in case client didn't provide some)
            acc = metrics.get("acc", float('nan'))
            roc_auc = metrics.get("roc_auc", float('nan'))
            pr_auc = metrics.get("pr_auc", float('nan'))
            val_loss = metrics.get("val_loss", float('nan'))

            print(f"[SERVER] Round {round_num} – Site {site_id} → "
                  f"Acc: {np.nan_to_num(acc):.4f}, AUC: {np.nan_to_num(roc_auc):.4f}, PR-AUC: {np.nan_to_num(pr_auc):.4f}, ValLoss: {np.nan_to_num(val_loss):.4f}")

            all_acc.append(acc)
            all_auc.append(roc_auc)
            all_pr.append(pr_auc)
            all_val_loss.append(val_loss)

        # === Rata-rata global (ignore NaNs using nanmean) ===
        mean_acc = np.nanmean(all_acc)
        mean_auc = np.nanmean(all_auc)
        mean_pr  = np.nanmean(all_pr)
        mean_val_loss = np.nanmean(all_val_loss)  # important: average of clients' val_loss

        epoch_time = time.time() - epoch_start
        cumulative_time = time.time() - cumulative_start

        print(f"[SERVER] Round {round_num} 🌍 GLOBAL AVG → "
              f"Acc: {np.nan_to_num(mean_acc):.4f}, AUC: {np.nan_to_num(mean_auc):.4f}, PR-AUC: {np.nan_to_num(mean_pr):.4f}, ValLoss: {np.nan_to_num(mean_val_loss):.4f}")

        print("================== ⏱ TIME MODEL ⏱ ==================")
        print(f"[SERVER] Consumption: {consumption_MB:.4f} MB, Agg Time: {agg_time:.4f}s, Epoch Time: {epoch_time:.4f}s, Cumulative Time: {cumulative_time:.2f}s")

        # --- TensorBoard logging ---
        writer.add_scalar("Global/Acc", mean_acc, round_num)
        writer.add_scalar("Global/AUC", mean_auc, round_num)
        writer.add_scalar("Global/PR_AUC", mean_pr, round_num)
        writer.add_scalar("Global/Val_Loss", mean_val_loss, round_num)
        for name, param in global_model.named_parameters():
            writer.add_histogram(f"Params/{name}", param.detach().cpu().numpy(), round_num)

        # --- Simpan ke CSV (human readable) ---
        with open(logfile, "a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                round_num,
                mean_acc,
                mean_auc,
                mean_pr,
                mean_val_loss,
                agg_time,
                init_time,
                consumption_MB,
                epoch_time,
                cumulative_time
            ])

        # --- Early stopping (pakai mean_val_loss) ---
        # Use mean_val_loss directly as the quantity to minimize. This matches
        # original centralized reference which averaged per-site val losses.
        # If mean_val_loss is NaN (e.g., all sites failed to produce val_loss),
        # skip early stopping update for safety.
        if not np.isnan(mean_val_loss):
            val_loss = float(mean_val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                best_path = f"agg/global_best_round{round_num}.pth"
                torch.save({
                    'round': round_num,
                    'state_dict': global_model.state_dict(),
                    'val_loss': best_val_loss
                }, best_path)
                print(f"[SERVER] ✅ Improvement: val_loss {val_loss:.4f} -> saved best model → {best_path}")
            else:
                if round_num >= 50:
                    bad_epochs += 1
                    print(f"[SERVER] ❌ No improvement: val_loss {val_loss:.4f}, patience {bad_epochs}/{patience}")
                else:
                    print(f"[SERVER] ⏩ No improvement (ignored, before epoch 50)")
        else:
            print("[SERVER] ⚠️ mean_val_loss is NaN — skipping early-stopping update for this round")

        if round_num >= 50 and bad_epochs >= patience:
            print(f"[SERVER] 🛑 Early stopping triggered at round {round_num}. Best val_loss={best_val_loss:.4f}")
            break

        total_run_time = time.time() - cumulative_start
        print(f"[SERVER] Training finished. ⏱ Total run time: {total_run_time:.2f}s")
            
    writer.close()
    print("[SERVER] Finished all rounds.")