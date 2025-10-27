import os
import torch

def count_model_params_from_state(state_dict):
    return sum(v.numel() for v in state_dict.values())

def load_total_params_from_pth(pth_path):
    state = torch.load(pth_path, map_location="cpu")
    return count_model_params_from_state(state)

def detect_encrypted_layers_from_files(base_path, base_pth_name):
    """
    base_path: folder agg/ (string)
    base_pth_name: filename without extension, e.g. "site0_round0" (string)
    returns: dict layer_name -> is_encrypted (True/False)
    """
    layer_encrypted = {}
    # scan files matching prefix
    prefix = os.path.join(base_path, base_pth_name)
    for fname in os.listdir(base_path):
        if not fname.startswith(os.path.basename(prefix)):
            continue
        # filenames like site0_round0_{layer}.bin or site0_round0_{layer}_part0.bin
        if fname.endswith(".bin"):
            # extract layer name heuristically
            parts = fname.replace(".bin","").split("_")
            # reconstruct layer name from last parts (depends on your naming)
            # e.g. site0_round0_conv1.weight_part0 -> layer key "conv1.weight"
            # This heuristic may need adjustment to your exact naming pattern.
            # We'll simply mark the full remainder as encrypted indicator.
            layer_key = "_".join(parts[2:])  # remove site0_round0 prefix
            layer_encrypted[layer_key] = True
    return layer_encrypted

def compute_ier_from_model_and_files(model_pth_path, agg_folder, upload_base_name):
    # total params in model state
    state = torch.load(model_pth_path, map_location="cpu")
    total = count_model_params_from_state(state)

    # detect encrypted layers (coarse)
    enc_layers = detect_encrypted_layers_from_files(agg_folder, upload_base_name)
    enc_params = 0
    plain_params = 0

    # map layer keys from state to whether encrypted exists (best-effort mapping)
    for k,v in state.items():
        # try match: in your code the bin names might contain the exact layer key
        # We'll check if any encrypted indicator contains the layer name k (or suffix)
        matched = False
        for enc_key in enc_layers.keys():
            if k.replace(".", "_") in enc_key or enc_key in k:
                matched = True
                break
        if matched:
            enc_params += v.numel()
        else:
            plain_params += v.numel()

    # fallback sanity: ensure sums equal total
    if enc_params + plain_params != total:
        # if mismatch, assume remainder is encrypted (or plaintext) based on mode
        # but let's just report totals and warn
        print("Warning: enc + plain != total ({}/{}/{})".format(enc_params, plain_params, total))

    ier = plain_params / total if total>0 else 0.0
    return {
        "total_params": total,
        "encrypted_params": enc_params,
        "plaintext_params": plain_params,
        "IER": ier
    }

if __name__ == "__main__":
    # EXAMPLE usage - sesuaikan nama file
    model_pth = "models/init_model.pth"          # path ke model reference (atau global server file)
    agg_folder = "agg"
    upload_base_name = "site0_round0"           # nama prefix upload file yang klien buat

    res = compute_ier_from_model_and_files(model_pth, agg_folder, upload_base_name)
    print(res)