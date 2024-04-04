import numpy as np
import time
from multiprocessing import Pool
from functools import partial
import itertools
import argparse

from ckks import (
    CKKSParameters, CKKSKeyGenerator, CKKSEncoder, CKKSEncryptor,
    CKKSDecryptor, CKKSEvaluator
)

WORKER_COUNT = 32

def create_random_complex_matrix(dimensions):
    return np.random.random((dimensions, dimensions)) + 1j * np.random.random((dimensions, dimensions))

def multiply_matrices_in_blocks(matrix, vector, ckks_keys, block_sz):
    n = matrix.shape[0]
    encrypted_vector_blocks = []
    
    # Encoding and encryption
    for start in range(0, n, block_sz):
        block = vector[start:start + block_sz]
        encoded_block = ckks_keys['encoder'].encode(block, ckks_keys['scale'])
        encrypted_block = ckks_keys['encryptor'].encrypt(encoded_block)
        encrypted_vector_blocks.append(encrypted_block)

    # Matrix multiplication
    result_blocks = []
    for i in range(n // block_sz):
        result_block = None
        for j in range(n // block_sz):
            sub_matrix = matrix[i*block_sz:(i+1)*block_sz, j*block_sz:(j+1)*block_sz]
            rotation_keys = {k: ckks_keys['keygen'].generate_rot_key(k) for k in range(len(sub_matrix))}
            
            partial_product = ckks_keys['evaluator'].multiply_matrix(encrypted_vector_blocks[j], sub_matrix, rotation_keys, ckks_keys['encoder'])
            result_block = partial_product if result_block is None else ckks_keys['evaluator'].add(result_block, partial_product)

        decrypted_block = ckks_keys['decryptor'].decrypt(result_block)
        result_blocks.extend(ckks_keys['encoder'].decode(decrypted_block))

    return result_blocks

def parallel_row_multiplication(matrix, vector, ckks_keys, block_sz, row_idx):
    n = matrix.shape[0]
    result_row = None
    for col_block_idx in range(n // block_sz):
        sub_matrix = matrix[row_idx*block_sz:(row_idx+1)*block_sz, col_block_idx*block_sz:(col_block_idx+1)*block_sz]
        rotation_keys = {k: ckks_keys['keygen'].generate_rot_key(k) for k in range(len(sub_matrix))}

        partial_product = ckks_keys['evaluator'].multiply_matrix(vector[col_block_idx], sub_matrix, rotation_keys, ckks_keys['encoder'])
        result_row = partial_product if result_row is None else ckks_keys['evaluator'].add(result_row, partial_product)

    decrypted_row = ckks_keys['decryptor'].decrypt(result_row)
    return ckks_keys['encoder'].decode(decrypted_row)

def run_parallel_matrix_multiplication(matrix, vector, ckks_keys, block_sz, row_based=True):
    n = matrix.shape[0]
    encrypted_vector_blocks = []
    
    for start in range(0, n, block_sz):
        block = vector[start:start + block_sz]
        encoded_block = ckks_keys['encoder'].encode(block, ckks_keys['scale'])
        encrypted_block = ckks_keys['encryptor'].encrypt(encoded_block)
        encrypted_vector_blocks.append(encrypted_block)

    with Pool(WORKER_COUNT) as pool:
        if row_based:
            result = pool.map(partial(parallel_row_multiplication, matrix, encrypted_vector_blocks, ckks_keys, block_sz),
                              range(n // block_sz))
        else:
            pass

    return list(itertools.chain.from_iterable(result))

def main():
    parser = argparse.ArgumentParser(description='Encrypted Matrix Multiplication')
    parser.add_argument("--block-size", type=int, default=16, help="Size of the blocks")
    parser.add_argument("--matrix-size", type=int, default=128, help="Size of the matrix")
    parser.add_argument("--algorithm", choices=['standard', 'parallel'], default='parallel', help="Algorithm choice")
    args = parser.parse_args()

    # Initialize parameters and keys
    poly_degree = 4096
    scale = 1 << 40
    ckks_params = CKKSParameters(poly_degree=poly_degree, ciph_modulus=1 << 600, big_modulus=1 << 1200, scaling_factor=scale)
    key_gen = CKKSKeyGenerator(ckks_params)
    ckks_keys = {
        'public_key': key_gen.public_key,
        'secret_key': key_gen.secret_key,
        'relin_key': key_gen.relin
