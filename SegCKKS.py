import math
import numpy as np
from Pyfhel import Pyfhel
import json
import warnings
warnings.filterwarnings('ignore')

with open('ModDict.json', 'r') as fcc_file:
    schemeDict = json.load(fcc_file)

def generate_ckks_key(sec_level, mul_depth, poly_moduls_degree):
    HE = Pyfhel()
    ckks_params = schemeDict.get(sec_level, {}).get(mul_depth, {}).get(poly_moduls_degree, {})
    print("sec_level:", sec_level, "mul_depth:", mul_depth, "poly_moduls_degree:", poly_moduls_degree)
    print("ckks_params: ", ckks_params)
    status = HE.contextGen(**ckks_params)
    print("success:" if status else "failed:", "valid" if status else "invalid")
    HE.keyGen()
    return HE


def enc_vector(HE, arr_x):
    arr_x = np.array(arr_x, dtype=np.float64)
    ptxt_x = HE.encodeFrac(arr_x)
    ctxt_x = HE.encryptPtxt(ptxt_x)
    return ctxt_x


def dec_vector(HE, ctxt_x):
    r_x = HE.decryptFrac(ctxt_x)
    _r = lambda x: np.round(x, decimals=6)
    return _r(r_x)


def seg_enc_vector(HE, vector, vecl):
    block_enc_arr = [] 
    block_len = HE.get_nSlots()
    block_arr_len = math.ceil(vecl / block_len)
    for i in range(block_arr_len):
        start_index = block_len * i
        end_index = min(block_len * (i+1), vecl)
        if end_index > vecl:
            end_index = vecl
        vector_block = np.array(vector[start_index:end_index], dtype=np.float64)
        enc_vector_block = enc_vector(HE, vector_block)
        block_enc_arr.append(enc_vector_block)
    return block_enc_arr


def seg_dec_vector(HE, block_enc_arr):
    dec_result = []
    for block_enc in block_enc_arr:
        dec_result.append(dec_vector(HE, block_enc))
    dec_result = np.concatenate(dec_result)
    return dec_result

def save_ckks_public(HE, pub_path="agg/ckks_pub.key", ctx_path="agg/ckks_context.con"):
    HE.save_context(ctx_path)
    HE.save_public_key(pub_path)

def load_ckks_public(pub_path, ctx_path="agg/ckks_context.con"):
    # HE = Pyfhel()
    # HE.load_context(ctx_path)  
    # HE.load_public_key(pub_path)
    # return HE
    HE = Pyfhel()
    with open(ctx_path, "rb") as f:
        HE.from_bytes_context(f.read())
    HE.load_public_key(pub_path)
    return HE

def add_cipher(HE, c1, c2):
    return HE.add(c1, c2)

# def scalar_mult_cipher(HE, c, scalar):
#     ptxt = HE.encodeFrac(np.array([scalar], dtype=np.float64))
#     return HE.multiply_plain(c, ptxt)

def scalar_mult_cipher(HE, c, scalar):
    # Encode scalar as full-slot vector to avoid ambiguous broadcasting / scale mismatch
    vec = np.full(HE.get_nSlots(), float(scalar), dtype=np.float64)
    ptxt = HE.encodeFrac(vec)
    return HE.multiply_plain(c, ptxt)

# def save_ckks_secret(HE, sec_path="keys/ckks.key"):
#     """Simpan secret key (hanya client yang punya)."""
#     HE.save_secret_key(sec_path)

def save_ckks_secret(HE, sec_path="agg/ckks_secret.key"):
    HE.save_secret_key(sec_path)
    # Also ensure context is saved (server/clients use same ctx)
    HE.save_context("agg/ckks_context.con")

def load_ckks_secret(sec_path="keys/ckks.key", ctx_path="agg/ckks_context.con"):
    """Load context + secret key (buat dekripsi di client)."""
    HE = Pyfhel()
    HE.load_context(ctx_path)       # harus sama dengan yg dipakai server
    HE.load_secret_key(sec_path)    # secret key private
    return HE
