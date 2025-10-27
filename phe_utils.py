# phe_utils.py
from phe import paillier
import torch

def fp_encode(x: torch.Tensor, scale: int):
    return torch.round(x * scale).to(torch.int64)

def fp_decode(x_int: torch.Tensor, scale: int):
    return (x_int.to(torch.float64) / scale).to(torch.float32)

def tensor_flatten_to_list_int(t: torch.Tensor, scale: int):
    x = fp_encode(t.detach().cpu(), scale)
    return x.view(-1).tolist(), x.size()

def list_int_to_tensor(lst, shape, device, scale):
    x = torch.tensor(lst, dtype=torch.int64).view(shape)
    return fp_decode(x, scale).to(device)

def encrypt_int_list(pubkey, ints):
    return [pubkey.encrypt(int(v)) for v in ints]

def sum_ciphertexts(cipher_lists):
    # elementwise sum of ciphertexts across sites
    summed = []
    for elems in zip(*cipher_lists):
        s = elems[0]
        for e in elems[1:]:
            s = s + e  # homomorphic addition
        summed.append(s)
    return summed

def decrypt_to_ints(privkey, cipher_list):
    return [privkey.decrypt(c) for c in cipher_list]

def approx_paillier_cipher_bytes(keybits):
    # Paillier ciphertext ~ 2 * modulus bytes (kasar)
    return (2 * (keybits // 8))