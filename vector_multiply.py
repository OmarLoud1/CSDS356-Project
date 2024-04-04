import numpy as np
from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_parameters import CKKSParameters

def create_complex_matrix(size):
    """Generate a square complex matrix with random values."""
    return np.random.rand(size, size) + 1j * np.random.rand(size, size)

def encrypt_vector(encoder, encryptor, vector, scale, block_size):
    """Encrypt blocks of the vector."""
    encrypted_blocks = []
    for i in range(0, len(vector), block_size):
        block = vector[i:i + block_size]
        encoded = encoder.encode(block, scale)
        encrypted = encryptor.encrypt(encoded)
        encrypted_blocks.append(encrypted)
    return encrypted_blocks

def matrix_vector_multiply(matrix, encrypted_vector, keys, block_size):
    """Multiply matrix by encrypted vector blocks and decrypt the results."""
    encoder, evaluator, decryptor = keys['encoder'], keys['evaluator'], keys['decryptor']
    n = matrix.shape[0]
    result = np.zeros(n, dtype=complex)

    for i in range(n // block_size):
        for j in range(n // block_size):
            block = matrix[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            rotation_keys = {k: keys['keygen'].generate_rot_key(k) for k in range(len(block))}
            product = evaluator.multiply_matrix(encrypted_vector[j], block, rotation_keys, encoder)
            if j == 0:
                aggregated_product = product
            else:
                aggregated_product = evaluator.add(aggregated_product, product)
        
        decrypted_block = decryptor.decrypt(aggregated_product)
        decoded_block = encoder.decode(decrypted_block)
        result[i * block_size:(i + 1) * block_size] = decoded_block

    return result

# Main parameters
MATRIX_SIZE = 16
BLOCK_SIZE = 16

# CKKS parameters initialization
params = CKKSParameters(poly_degree=BLOCK_SIZE *2, ciph_modulus=1 << 600, big_modulus=1 << 1200, scaling_factor=1 << 30)
keygen = CKKSKeyGenerator(params)
keys = {
    'encoder': CKKSEncoder(params),
    'encryptor': CKKSEncryptor(params, keygen.public_key, keygen.secret_key),
    'decryptor': CKKSDecryptor(params, keygen.secret_key),
    'evaluator': CKKSEvaluator(params),
    'keygen': keygen
}

# Generating matrix and vector, and performing encrypted multiplication
matrix = create_complex_matrix(MATRIX_SIZE)
vector = create_complex_matrix(MATRIX_SIZE)[0]  # Using the first row as a vector
encrypted_vector = encrypt_vector(keys['encoder'], keys['encryptor'], vector, params.scaling_factor, BLOCK_SIZE)
result = matrix_vector_multiply(matrix, encrypted_vector, keys, BLOCK_SIZE)

print( "\n", vector, "\n", matrix @ vector)
print("Encrypted matrix multiplication result:", result)
