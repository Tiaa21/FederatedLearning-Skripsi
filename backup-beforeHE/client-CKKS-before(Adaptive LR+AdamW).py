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
from Pyfhel import PyCtxt, Pyfhel, PyPtxt
import shutil
from SegCKKS import generate_ckks_key, save_ckks_public, enc_vector, dec_vector, seg_enc_vector, seg_dec_vector, load_ckks_secret, save_ckks_secret
import re

EPS = 1e-10

os.makedirs("keys", exist_ok=True)

# === timing init client model ===
init_start = time.time()
init_time = time.time() - init_start

# =====================================================
# Parser dengan CKKS args ala FedSHE
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--site_id", type=int, required=True)
parser.add_argument('--ckks_sec_level', type=str, default='128',
                    help="CKKS security level: 128,192,256")
parser.add_argument('--ckks_mul_depth', type=str, default='0',
                    help="CKKS multiplication depth: 0,1,2,3,4")
parser.add_argument('--ckks_key_len', type=str, default='1024',
                    help="CKKS poly modulus degree / slot length")
args = parser.parse_args()

# =====================================================
# Generate Key CKKS (Public dan Private)
# =====================================================

# HE = None
# if params.mode == "CKKS":
#     HE = generate_ckks_key(
#         sec_level=args.ckks_sec_level,
#         mul_depth=args.ckks_mul_depth,
#         poly_moduls_degree=args.ckks_key_len
#     )
#     HE.save_secret_key(f"keys/ckks_site{args.site_id}.key")
#     HE.save_context("agg/ckks_context.con")
#     save_ckks_public(HE, f"agg/ckks_pub_site{args.site_id}.key")
#     print(f"[CLIENT {args.site_id}] ✅ Generated CKKS keys and saved public key.")

HE = None
if params.mode == "CKKS":
    ctx_path = "agg/ckks_context.con"
    pub_path = "agg/ckks_pub.key"
    sec_path = "agg/ckks_secret.key"

    # site0 will act as KMC if keys not present
    if args.site_id == 0 and not (os.path.exists(ctx_path) and os.path.exists(pub_path) and os.path.exists(sec_path)):
        HE = generate_ckks_key(
            sec_level=args.ckks_sec_level,
            mul_depth=args.ckks_mul_depth,
            poly_moduls_degree=args.ckks_key_len
        )
        # Save context + public + secret for distribution
        HE.save_context(ctx_path)
        save_ckks_public(HE, pub_path)
        save_ckks_secret(HE, sec_path)
        print(f"[CLIENT {args.site_id}] ✅ Generated CKKS keys & saved context/public/secret to agg/")
    else:
        # other clients (and site0 if keys already exist) wait until files are available
        waited = False
        while not (os.path.exists(ctx_path) and os.path.exists(pub_path) and os.path.exists(sec_path)):
            if not waited:
                print(f"[CLIENT {args.site_id}] Menunggu ckks context/pub/secret ...")
                waited = True
            time.sleep(1)
        # load context + public + secret (shared)
        HE = Pyfhel()
        with open(ctx_path, "rb") as f:
            HE.from_bytes_context(f.read())
        HE.load_public_key(pub_path)
        HE.load_secret_key(sec_path)
        print(f"[CLIENT {args.site_id}] ✅ Loaded CKKS context, public and secret (shared).")

# =====================================================
# Function Encrypt dan Decrypt Client
# =====================================================

def compute_comm_ckks(model, N, D):
    """
    Hitung komunikasi total (bit → MB) untuk model setelah adopsi CKKS.
    Rumus: COMM_CKKS = sum_i ceil(2 * L_i / N) * N * 2 * D * 64 bit
    """
    total_bits = 0
    for name, param in model.state_dict().items():
        if param.numel() == 0:
            continue
        Li = param.numel()
        blocks = np.ceil((2 * Li) / N)
        total_bits += blocks * N * 2 * D * 64

    total_MB = total_bits / (8 * 1024 * 1024)
    return total_MB

def encrypt_update(w_old, w_new):
    update_w = {}
    for k in w_new.keys():

        # --- skip param yang tidak bisa dienkripsi ---
        if any(s in k for s in ["running_mean", "running_var", "num_batches_tracked"]):
            print(f"[CLIENT {args.site_id}] ⚠️ Skipping {k} (BN stats)")
            continue

        # ambil perbedaan weight dan ubah ke numpy float64
        diff = (w_new[k] - w_old[k]).detach().cpu().view(-1).numpy().astype(np.float64)

        if diff.size == 0:
            print(f"[CLIENT {args.site_id}] ⚠️ Skipping {k} (empty tensor)")
            continue

        # pilih direct vs segmented encryption
        if len(diff) <= HE.get_nSlots():
            update_w[k] = enc_vector(HE, diff)
            print(f"[CLIENT {args.site_id}] 🔒 Encrypted {k} (direct)")
        else:
            update_w[k] = seg_enc_vector(HE, diff, len(diff))
            print(f"[CLIENT {args.site_id}] 🔒 Encrypted {k} (segmented, {len(diff)} values)")
                
    return update_w

def strip_epoch_prefix(key: str):
    clean_key = key
    # buang angka depan
    if "_" in key and key.split("_", 1)[0].isdigit():
        clean_key = key.split("_", 1)[1]
    # buang .bin
    if clean_key.endswith(".bin"):
        clean_key = clean_key[:-4]
    return clean_key

def decrypt_and_load_model(model, enc_path, sec_path=None, ctx_path=None, pub_path=None):
    HE = Pyfhel()
    import re
    
    # 1️⃣ Load CKKS context
    if ctx_path is not None:
        with open(ctx_path, "rb") as f:
            HE.from_bytes_context(f.read())
    else:
        HE.contextGen(scheme='CKKS', n=4096, scale=2**20, qi_sizes=[30,20,30])
        HE.relinKeyGen()
    
    # 2️⃣ Load secret/public key
    if pub_path is not None:
        HE.load_public_key(pub_path)
    if sec_path is not None:
        HE.load_secret_key(sec_path)
    
    # 3️⃣ Load serialized bytes
    enc_state = torch.load(enc_path, map_location="cpu")
    
    new_state = {}
    segment_buffer = {}

    # 🔹 Decode semua ciphertext
    for raw_k, v in enc_state.items():
        k = strip_epoch_prefix(raw_k)
        # print(f"[DEBUG] {raw_k} -> type(v): {type(v)}")        
        if isinstance(v, list):  # segmented ciphertext
            ctxt_list = [PyCtxt(pyfhel=HE, bytestring=b) for b in v]
            # print(f"    Segmented, len={len(v)}, elem_type={type(v[0])}")
            dec = seg_dec_vector(HE, ctxt_list)
            tensor = torch.tensor(dec, dtype=torch.float32)
            # masukkan ke buffer
            if k not in segment_buffer:
                segment_buffer[k] = []
            segment_buffer[k].append(tensor)
        
        else:  # single ciphertext
            print(f"    Single, elem_type={type(v)}")

    # 🔹 Gabungkan segmented parts
    for k, parts in segment_buffer.items():

        if k.endswith(".bin"):
            k = k[:-4]

        if k in model.state_dict():
            expected_shape = model.state_dict()[k].shape
            expected_numel = model.state_dict()[k].numel()
    
            full_tensor = torch.cat(parts, dim=0)
    
            if full_tensor.numel() > expected_numel:
                full_tensor = full_tensor[:expected_numel]
    
            elif full_tensor.numel() < expected_numel:
                pad_size = expected_numel - full_tensor.numel()
                full_tensor = torch.cat([full_tensor, torch.zeros(pad_size, dtype=full_tensor.dtype)], dim=0)
    
            new_state[k] = full_tensor.view(expected_shape)

            # Bentuk delta ke ukuran layer
            delta_tensor = full_tensor.view(expected_shape).to(params.device)

            # 🔥 Apply delta: W_new = W_old + delta
            with torch.no_grad():
                current = model.state_dict()[k]
                model.state_dict()[k].copy_(current.to(delta_tensor.device) + delta_tensor)

        else:
            print(f"[Missing segmented key] {k}")

    print("Model keys:", list(model.state_dict().keys())[:10])
    print("New state keys:", list(new_state.keys())[:10])

    # model.load_state_dict(new_state, strict=False)
    # model.load_state_dict(new_state, strict=True)
    # return model
    return model

    
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--site_id", type=int, required=True)
    # args = parser.parse_args()
    
    site_id = args.site_id

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
                "Encrypt Time", 
                "Decrypt Time",
                "Transmission Time",
                "COMM_CKKS(MB)"
            ])

    log_dir = os.path.join("runs", f"site{args.site_id}")

    # # safety check: recreate dir if missing
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir, exist_ok=True)

    # # wait sebentar untuk Windows filesystem (bug TensorBoard)
    # time.sleep(0.5)

    # # TensorBoard writer
    writer = SummaryWriter(log_dir)
    print(f"[CLIENT {args.site_id}] ✅ TensorBoard logging to: {log_dir}")

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
    # optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    # try:
    #     optimizerG = optim.Adam(model.encoder.parameters(), lr=params.learning_rate)
    # except Exception:
    #     optimizerG = optim.Adam(model.parameters(), lr=params.learning_rate)
    # optimizerD = optim.Adam(disc.parameters(), lr=params.learning_rate)

    # optimizers
    optimizer = optim.SGD(
        model.parameters(),
        lr=params.learning_rate,       # misal 5e-6
        momentum=0.9,
        weight_decay=1e-4
    )
    
    try:
        optimizerG = optim.SGD(
            model.encoder.parameters(),
            lr=params.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
    except Exception:
        optimizerG = optim.SGD(
            model.parameters(),
            lr=params.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        
    optimizerD = optim.SGD(
        model.parameters(),
        lr=params.learning_rate,       # misal 5e-6
        momentum=0.9,
        weight_decay=1e-4
    )

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
        print(f"[CLIENT {args.site_id}] 🛠 TRAINING START")
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

        # --- Simpan model lama ---
        w_old = {k: v.clone().detach() for k,v in model.state_dict().items()}

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

        # === Adversarial Alignment ===
        global_model = None
        adv_global_plain_path = f"agg/global_round{max(0, round_num-1)}.pth"       # plain expected
        adv_global_enc_path   = f"agg/global_round{max(0, round_num-1)}_enc.pth"   # ckks expected

        # prefer plain if exists (server may store both)
        if os.path.exists(adv_global_plain_path):
            try:
                state = torch.load(adv_global_plain_path, map_location=device)
                global_model = Classifier().to(device)
                global_model.load_state_dict(state)
                global_model.eval()
                print(f"[CLIENT {args.site_id}] ADV: loaded plain global model {adv_global_plain_path}")
            except Exception as e:
                print(f"[CLIENT {args.site_id}] ADV: failed to load plain global model: {e}")
                global_model = None
        else:
            # if plain not present and we're in CKKS scenario, try to decrypt aggregated ciphertext
            if params.mode == "CKKS" and os.path.exists(adv_global_enc_path):
                try:
                    # Use your decrypt_and_load_model but adapt to return a model instance
                    tmp_model = Classifier().to(device)
                    tmp_model = decrypt_and_load_model(tmp_model, adv_global_enc_path, sec_path="agg/ckks_secret.key", ctx_path="agg/ckks_context.con", pub_path="agg/ckks_pub.key")
                    tmp_model.eval()
                    global_model = tmp_model
                    print(f"[CLIENT {args.site_id}] ADV: decrypted and loaded CKKS aggregated model {adv_global_enc_path}")
                except Exception as e:
                    print(f"[CLIENT {args.site_id}] ADV: failed to decrypt/load CKKS aggregated model: {e}")
                    global_model = None
            else:
                global_model = None

        print(f"[CLIENT {args.site_id}] Check ADV: round={round_num}, global_model={global_model is not None}, n_epochs_adv={getattr(params, 'n_epochs_adversarial', 0)}")

        # global_path = f"agg/global_round{max(0, round_num-1)}.pth"
        # if os.path.exists(global_path):
        #     try:
        #         global_state = torch.load(global_path, map_location=device)
        #         global_model = Classifier().to(device)
        #         global_model.load_state_dict(global_state)
        #         global_model.eval()
        #     except Exception as e:
        #         print(f"[CLIENT {args.site_id}] Failed to load global model for adv: {e}")
        #         global_model = None
        # else:
        #     global_model = None
            
        # print(f"[CLIENT {args.site_id}] Check ADV: round={round_num}, global_model={global_model is not None}, n_epochs_adv={getattr(params, 'n_epochs_adversarial', 0)}")

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

        local_train_time = time.time() - train_start

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

        # --- Prepare update (plain delta or ciphertext) ---
        w_new = {k: v.clone().detach() for k, v in model.state_dict().items()}

        enc_time = 0.0
        dec_time = 0.0

        if params.mode == "Plain":
            # delta update
            update_w = {k: (w_new[k] - w_old[k]) for k in w_new.keys()}
        elif params.mode == "CKKS":
            # encrypt update (menghasilkan struktur ciphertext per-layer)
            enc_start = time.time()
            update_w = encrypt_update(w_old, w_new)  # implementasi kamu harus mengembalikan per-layer ciphertext / segmented list
            enc_time = time.time() - enc_start
            print(f"[CLIENT {args.site_id}] ⏱ Encrypt time: {enc_time:.4f}s")

            # === Communication Cost CKKS (theoretical) ===
            comm_ckks_MB = compute_comm_ckks(model, int(args.ckks_key_len), int(args.ckks_mul_depth) if int(args.ckks_mul_depth)>0 else 1)
            print(f"[CLIENT {args.site_id}] 💬 Theoretical COMM_CKKS: {comm_ckks_MB:.4f} MB")
        else:
            update_w = w_new

        # === UPLOAD ===
        local_path = f"agg/site{args.site_id}_round{round_num}.pth"

        # Untuk Plain: simpan state_dict atau delta
        try:
            if params.mode == "Plain":
                # menyimpan aktual weights (atau bisa juga menyimpan delta tergantung server)
                torch.save(w_new, local_path)
            elif params.mode == "CKKS":
                # simpan placeholder pth agar server tahu upload selesai + simpan ciphertext & context sebagai .bin
                torch.save({k: None for k in update_w.keys()}, local_path)

                # simpan context sekali (kamu bisa skip ini untuk site != 0 jika re-use context)
                ctx_path = local_path.replace(".pth", "_ctx.bin")
                with open(ctx_path, "wb") as f:
                    f.write(HE.to_bytes_context())

                # simpan ciphertext per layer
                for k, v in update_w.items():
                    if isinstance(v, list):  # segmented ciphertext
                        for i, part in enumerate(v):
                            part_path = local_path.replace(".pth", f"_{k}_part{i}.bin")
                            with open(part_path, "wb") as f:
                                f.write(part.to_bytes())
                    else:
                        part_path = local_path.replace(".pth", f"_{k}.bin")
                        with open(part_path, "wb") as f:
                            f.write(v.to_bytes())
            else:
                torch.save(w_new, local_path)
        except Exception as e:
            print(f"[CLIENT {args.site_id}] ⚠️ Error saving upload files: {e}")

        print(f"[CLIENT {args.site_id}] Uploaded round {round_num}")

        # hitung size upload (Plain: ukuran pth, CKKS: jumlah file)
        try:
            if params.mode == "CKKS":
                upload_size = 0.0
                # include the .pth, .ctx.bin and all _{k}.bin/_part files
                upload_size += os.path.getsize(local_path)
                ctx_path = local_path.replace(".pth", "_ctx.bin")
                if os.path.exists(ctx_path):
                    upload_size += os.path.getsize(ctx_path)
                # per-layer files
                for fname in os.listdir(os.path.dirname(local_path) or "."):
                    if fname.startswith(os.path.basename(local_path).replace(".pth", "")) and fname.endswith(".bin"):
                        upload_size += os.path.getsize(os.path.join(os.path.dirname(local_path), fname))
                upload_size = upload_size / (1024.0 * 1024.0)
            else:
                upload_size = os.path.getsize(local_path) / (1024.0 * 1024.0)
        except Exception:
            upload_size = -1.0

        # === Tunggu global sync ===
        if params.mode == "Plain":
            next_global = f"agg/global_round{round_num}.pth"
        else:
            # server diharapkan menyimpan hasil agregasi ciphertext di file ini
            next_global = f"agg/global_round{round_num}_enc.pth"

        comm_start = time.time()
        while not os.path.exists(next_global):
            time.sleep(2)

        # hitung ukuran download
        try:
            if params.mode == "CKKS":
                # pastikan file yang server hasilkan (ciphertext aggregated) memang next_global
                download_size = os.path.getsize(next_global) / (1024.0 * 1024.0)
            else:
                download_size = os.path.getsize(next_global) / (1024.0 * 1024.0)
        except Exception:
            download_size = -1.0

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
                enc_time if params.mode == "CKKS" else 0.0,
                dec_time if params.mode == "CKKS" else 0.0,
                comm_time,
                comm_ckks_MB if params.mode == "CKKS" else 0.0
            ])

        print(f"================== 🖥 MODEL {round_num} ==================")
        print(f"[CLIENT {args.site_id}] Upload: {upload_size:.4f} MB, Download: {download_size:.4f} MB")
        print(f"CLIENT {args.site_id}] ⏱ Training Time: {local_train_time:.4f}s, ⏱ Transmission time (T_trans): {comm_time:.4f}s")
        

        while True:
            try:
                # --- Apply global model ---
                if params.mode == "Plain":
                    state_dict = torch.load(next_global, map_location=params.device)
                    model.load_state_dict(state_dict)
                elif params.mode == "CKKS":
                    # gunakan decrypt_and_load_model yang meng-handle ciphertext aggregated file
                    ctx_path = "agg/ckks_context.con"   # path konteks yang server share atau kamu simpan sebelumnya
                    sec_path = f"agg/ckks_secret_site{args.site_id}.key" if os.path.exists(f"agg/ckks_secret_site{args.site_id}.key") else "agg/ckks_secret.key"
                    pub_path = "agg/ckks_pub.key"
                    dec_start = time.time()
                    model = decrypt_and_load_model(model, next_global, sec_path, ctx_path, pub_path)
                    dec_time = time.time() - dec_start
                    print(f"[CLIENT {args.site_id}] ⏱ Decrypt time: {dec_time:.4f}s")
                break
            except Exception as e:
                print(f"[CLIENT {args.site_id}] Global file invalid or decryption failed, retrying... ({e})")
                time.sleep(2)


        print(f"[CLIENT {args.site_id}] Synced global round {round_num}")

        round_num += 1