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
from Pyfhel import Pyfhel, PyCtxt, PyPtxt
from SegCKKS import load_ckks_public, add_cipher, scalar_mult_cipher
from collections import defaultdict
import re

# torch.serialization.add_safe_globals([Pyfhel, PyCtxt, PyPtxt])

best_val_loss = float("inf")
patience = getattr(params, "patience", 10)  # bisa set di params.py
bad_epochs = 0

FRESH_DIRS = ["agg", "runs"]

for d in FRESH_DIRS:
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
            print(f"Folder {d} berhasil dihapus!")
        except OSError as e:
            print(f"⚠️ Gagal hapus {d}: {e}, coba hapus manual.")
    os.makedirs(d, exist_ok=True)
    print(f"Folder {d} berhasil dibuat ulang")

# === PRE-CREATE subfolder di runs ===
subdirs = ["server", "site0", "site1", "site2"]
for sd in subdirs:
    os.makedirs(os.path.join("runs", sd), exist_ok=True)
print("📁 Pre-created:", ", ".join(os.path.join("runs", sd) for sd in subdirs))

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

# =====================================================
# Aggregation with noise
# =====================================================
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

# =====================================================
# CKKS aggregation with optional noise
# =====================================================
# def aggregate_ckks(weights_list, noise=0.0):
#     grouped = defaultdict(list)
#     for k in weights_list[0].keys():
#         if "_part" in k:
#             base, part = k.split("_part")
#             idx = int(part.split(".")[0])  # ambil index part
#             grouped[base].append((idx, k))
#         else:
#             grouped[k].append((None, k))

#     update_w_avg = {}
#     for base, parts in grouped.items():
#         if parts[0][0] is not None:  # segmented
#             parts = sorted(parts, key=lambda x: x[0])
#             csum = [weights_list[0][k] for _, k in parts]
#             for i in range(1, len(weights_list)):
#                 for j, (_, k) in enumerate(parts):
#                     csum[j] = add_cipher(HE_pub_list[i], csum[j], weights_list[i][k])
#             update_w_avg[base] = [scalar_mult_cipher(HE_pub_list[0], c, 1.0/len(weights_list)) for c in csum]
#         else:  # single
#             k = parts[0][1]
#             csum = weights_list[0][k]
#             for i in range(1, len(weights_list)):
#                 csum = add_cipher(HE_pub_list[i], csum, weights_list[i][k])
#             update_w_avg[base] = scalar_mult_cipher(HE_pub_list[0], csum, 1.0/len(weights_list))

#     return update_w_avg

# def strip_epoch_prefix(key: str):
#     clean_key = key
#     # buang angka depan
#     if "_" in key and key.split("_", 1)[0].isdigit():
#         clean_key = key.split("_", 1)[1]
#     # buang .bin
#     if clean_key.endswith(".bin"):
#         clean_key = clean_key[:-4]
#     return clean_key

def strip_epoch_prefix(key: str):
    clean_key = key
    # buang angka depan sebelum underscore (misal: "0_encoder..." → "encoder...")
    if "_" in key and key.split("_", 1)[0].isdigit():
        clean_key = key.split("_", 1)[1]
    return clean_key  # JANGAN hapus .bin di sini!

def aggregate_ckks(weights_list, HE_pub, noise=0.0):
    grouped = defaultdict(list)

    # Ambil semua key yang muncul di semua client (intersection)
    all_keys = set(weights_list[0].keys())
    for w in weights_list[1:]:
        all_keys &= set(w.keys())

    # for k in weights_list[0].keys():
    #     clean_k = strip_epoch_prefix(k)
    #     if "_part" in clean_k:
    #         base, parts = clean_k.split("_part")

    #         idx = int(parts.split(".bin")[0])
    #         grouped[base].append((idx, k))
    #     else:
    #         grouped[clean_k].append((None, k))

    for k in all_keys:  # hanya pakai key yang ada di semua client
        clean_k = strip_epoch_prefix(k)
        if clean_k.endswith(".bin"):
            clean_k = clean_k[:-4]

        if "_part" in clean_k:
            base, part_info = clean_k.split("_part")
            idx = int(part_info.split(".bin")[0])
            grouped[base].append((idx, k))
        else:
            grouped[clean_k].append((None, k))

    missing = [k for k in weights_list[0].keys() if k not in all_keys]
    if missing:
        print(f"⚠️ Warning: {len(missing)} keys missing in other clients, skipping them.")

    # cek hasil grouping (opsional, debugging)
    for base, v in grouped.items():
        print(f"{base}: {len(v)} parts")

    update_w_avg = {}
    min_parts = min(len(parts) for parts in grouped.values())
    for base, parts in grouped.items():
        parts = sorted(parts, key=lambda x: x[0])[:min_parts]
        if parts[0][0] is not None:  # segmented
            parts = sorted(parts, key=lambda x: x[0])
            csum = [weights_list[0][k] for _, k in parts]
            for i in range(1, len(weights_list)):
                for j, (_, k) in enumerate(parts):
                    csum[j] = add_cipher(HE_pub, csum[j], weights_list[i][k])
            update_w_avg[base] = [scalar_mult_cipher(HE_pub, c, 1.0/len(weights_list)) for c in csum]
        else:  # single
            k = parts[0][1]
            csum = weights_list[0][k]
            for i in range(1, len(weights_list)):
                csum = add_cipher(HE_pub, csum, weights_list[i][k])
            update_w_avg[base] = scalar_mult_cipher(HE_pub, csum, 1.0/len(weights_list))

    return update_w_avg

# =====================================================
# compute plain average for small key
# =====================================================

def compute_plain_average(weights_list, key_to_check):
    try:
        arrs = []
        for st in weights_list:
            # assume st[k] may be tensor or None
            if key_to_check in st:
                v = st[key_to_check]
                if isinstance(v, torch.Tensor):
                    arrs.append(v.view(-1).cpu().numpy())
        if len(arrs) > 0:
            plain_avg = np.mean(np.stack(arrs, axis=0), axis=0)
            return plain_avg
    except Exception:
        pass
    return None


# =====================================================
# Metrics
# =====================================================

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--eval_site", type=int, default=0, help="site ID untuk evaluasi global (ignored in multi-site)")  
    args = parser.parse_args()

    # === Load CKKS public key jika mode CKKS ===
    HE_pub_list = []
    for i in range(args.num_clients):
        # pubkey_path = f"agg/ckks_pub_site{i}.key"
        pubkey_path = "agg/ckks_pub.key"
        waited = False

        while not os.path.exists(pubkey_path):
            if not waited:
                # print(f"[SERVER] Menunggu public key dari client {i}...")
                print("[SERVER] Menunggu ckks public key & context dari KMC/site0...")
                waited = True
            time.sleep(2)
        # HE_pub_list.append(load_ckks_public(pubkey_path, "agg/ckks_context.con"))
        HE_pub = load_ckks_public(pubkey_path, "agg/ckks_context.con")
    
        print("[SERVER] ✅ CKKS public key & context loaded (single global key).")

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
            warned = False
            
            while True:
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    break
                else:
                    if not warned:
                        print(f"[SERVER] Menunggu file {local_path} (belum tersedia / masih kosong)...")
                        warned = True
                    time.sleep(2)
                    
            try:
                if local_path.endswith(".pth"):
                    # --- LOAD STATE_DICT NORMAL ---
                    state_dict = torch.load(local_path, map_location="cpu")
                    # weights_list.append(state_dict)
                    print(f"[SERVER] Loaded {local_path} successfully!")
            
                    # load CKKS context
                    ctx_file = local_path.replace(".pth", "_ctx.bin")
                    if os.path.exists(ctx_file):
                        HE = Pyfhel()
                        with open(ctx_file, "rb") as f:
                            HE.from_bytes_context(f.read())
                        print(f"[SERVER] ✅ CKKS context loaded for {local_path}")
                    
                        # load ciphertext files per layer
                        base_name = os.path.basename(local_path).replace(".pth", "")
                        ct_files = [f for f in os.listdir("agg") if f.startswith(base_name) and f.endswith(".bin") and "_ctx" not in f]
                    
                        encrypted_weights = {}
                        for cf in ct_files:
                            with open(os.path.join("agg", cf), "rb") as f:
                                b = f.read()
                            key_norm = cf.split("_round", 1)[-1]
                            key_norm = re.sub(r'^\d+_', '', key_norm)
                            # PyCtxt wrapper using server HE_pub
                            ctxt = PyCtxt(pyfhel=HE, bytestring=b)
                            encrypted_weights[key_norm] = ctxt
                            #     ctxt = PyCtxt(pyfhel=HE, bytestring=f.read())

                            # # NORMALIZE KEY: hilangkan prefix siteX_roundY_
                            # key_norm = cf.split("_round", 1)[-1]   # ambil mulai dari setelah "_round"
                            # encrypted_weights[key_norm] = ctxt
                        print(f"[SERVER] ✅ Loaded {len(encrypted_weights)} ciphertext parts for {local_path}")
                        weights_list.append(encrypted_weights)
            
            except EOFError:
                print(f"[ERROR] File {local_path} kosong/corrupted. Tunggu client upload ulang.")
                time.sleep(2)
                raise
            except Exception as e:
                print(f"[ERROR] Gagal load {local_path}: {e}. Tunggu/cek file.")
                time.sleep(2)
                raise


        # === Aggregasi with noise ===
        w = {i: 1.0 / args.num_clients for i in range(args.num_clients)}

        agg_start = time.time()

        if params.mode == "Plain":
            avg_state = aggregate_with_noise(
                weights_list,
                noise=getattr(params, 'noise', 0.0),
                noise_type=getattr(params, 'noise_type', 'G'),
                device=params.device
            )
            # Plain: avg_state is a state_dict (tensors)
            global_model.load_state_dict(avg_state)
            # save atomic
            global_path = f"agg/global_round{round_num}.pth"
            tmp_path = global_path + ".tmp"
            torch.save(global_model.state_dict(), tmp_path)
            os.replace(tmp_path, global_path)
            # also save plaintext aggregate for clients
        elif params.mode == "CKKS":
            avg_state_enc = aggregate_ckks(weights_list, HE_pub, noise=getattr(params,'noise',0.0))
            # avg_state_enc is dict of ciphertexts (PyCtxt or list of PyCtxt)
            # save ciphertext aggregate for clients to download
            # Convert PyCtxt / list of PyCtxt ke bytes
            serializable_state = {}
            for k, v in avg_state_enc.items():
                if isinstance(v, list):  # segmented ciphertext
                    serializable_state[k] = [ctxt.to_bytes() for ctxt in v]
                else:  # single ciphertext
                    serializable_state[k] = [v.to_bytes()]
            
            enc_path = f"agg/global_round{round_num}_enc.pth"
            tmp_enc = enc_path + ".tmp"
            torch.save(serializable_state, tmp_enc)
            os.replace(tmp_enc, enc_path)
            print(f"[SERVER] Round {round_num} aggregated → {enc_path}")
        else:
            raise ValueError("Unknown mode!")


        agg_time = time.time() - agg_start

        # global_model.load_state_dict(avg_state)

        # debug NaNs
        for name, param in global_model.state_dict().items():
            if torch.isnan(param).any():
                print(f"[DEBUG] NaN detected in layer {name}")

        # # === Save global model so clients can sync ===
        # global_path = f"agg/global_round{round_num}.pth"
        # tmp_path = global_path + ".tmp"
        # torch.save(global_model.state_dict(), tmp_path)
        # os.replace(tmp_path, global_path)  # atomic rename
        # print(f"[SERVER] Round {round_num} aggregated → {global_path}")

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
        consumption_bytes = sum(os.path.getsize(f"agg/site{i}_round{round_num}.pth") 
                        for i in range(args.num_clients))
        consumption_MB = consumption_bytes / (1024.0*1024.0)

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
                best_acc = mean_acc
                best_auc = mean_auc
                best_pr = mean_pr
                best_round = round_num
                bad_epochs = 0
                best_path = f"agg/global_best_round{round_num}.pth"
                torch.save({
                    'round': round_num,
                    'state_dict': global_model.state_dict(),
                    'val_loss': best_val_loss,
                    'acc': best_acc,
                    'auc': best_auc,
                    'pr_auc': best_pr
                }, best_path)
                print(f"[SERVER] 💾 Saved best model → {best_path}")
                print(f"[SERVER] ✅ Improvement: val_loss {best_val_loss:.4f} → {val_loss:.4f} | Acc: {mean_acc:.4f}")
            else:
                if round_num >= 20:
                    bad_epochs += 1
                    print(f"[SERVER] ⏳ Patience: {bad_epochs}/{patience}")
                    print(f"[SERVER] ❌ No improvement: (current {val_loss:.4f} ≥ best {best_val_loss:.4f} at round {best_round})")
                    print(f"[SERVER] 🧠 Best so far → ValLoss={best_val_loss:.4f}, Acc={best_acc:.4f}, AUC={best_auc:.4f}, PR-AUC={best_pr:.4f} (Round {best_round})")
                else:
                    print(f"[SERVER] ⏩ No improvement (current {val_loss:.4f} ≥ best {best_val_loss:.4f} at round {best_round})")
                    print(f"[SERVER] 🧠 Best so far → ValLoss={best_val_loss:.4f}, Acc={best_acc:.4f}, AUC={best_auc:.4f}, PR-AUC={best_pr:.4f} (Round {best_round})")
        else:
            print("[SERVER] ⚠️ mean_val_loss is NaN — skipping early-stopping update for this round")

        if round_num >= 20 and bad_epochs >= patience:
            print(f"[SERVER] 🛑 Early stopping triggered at round {round_num}. Best val_loss={best_val_loss:.4f}")
            print(f"[SERVER] 🏆 Best Model (Round {best_round}) → "f"Acc: {best_acc:.4f}, AUC: {best_auc:.4f}, PR-AUC: {best_pr:.4f}, ValLoss: {best_val_loss:.4f}")
            break

        total_run_time = time.time() - cumulative_start
        print(f"[SERVER] Training finished. ⏱ Total run time: {total_run_time:.2f}s")
            
    writer.close()
    print("[SERVER] Finished all rounds.")