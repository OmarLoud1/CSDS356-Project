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

def dot_product_encrypted(evaluator, vector1, vector2, relin_key):
    """Compute dot product of two encrypted vectors."""
    products = [evaluator.multiply(c1, c2, relin_key) for c1, c2 in zip(vector1, vector2)]
    dot_product = products[0]
    for product in products[1:]:
        dot_product = evaluator.add(dot_product, product)
    return dot_product

def matrix_multiply(matrix1, matrix2):
    """Multiply two plaintext matrices."""
    return np.dot(matrix1, matrix2)

def main():
    MATRIX_SIZE = 4
    BLOCK_SIZE = 4
    SCALING_FACTOR = 1 << 30

    # CKKS parameters initialization
    params = CKKSParameters(poly_degree=BLOCK_SIZE * 2, ciph_modulus=1 << 600, big_modulus=1 << 1200, scaling_factor=SCALING_FACTOR)
    keygen = CKKSKeyGenerator(params)
    keys = {
        'encoder': CKKSEncoder(params),
        'encryptor': CKKSEncryptor(params, keygen.public_key, keygen.secret_key),
        'decryptor': CKKSDecryptor(params, keygen.secret_key),
        'evaluator': CKKSEvaluator(params),
        'keygen': keygen
    }

    evaluator = keys['evaluator']
    relin_key = keys['keygen'].relin_key

    # Generating user-item ratings matrix
    user_item_ratings = np.random.randint(0, 6, size=(MATRIX_SIZE, MATRIX_SIZE))

    # Plaintext matrix multiplication
    plaintext_result = matrix_multiply(user_item_ratings, user_item_ratings.T)

    # Encrypt user-item ratings matrix
    encrypted_ratings_matrix = [encrypt_vector(keys['encoder'], keys['encryptor'], row, SCALING_FACTOR, BLOCK_SIZE) for row in user_item_ratings]

    # Encrypted matrix multiplication (using dot product)
    encrypted_result = []
    for row1 in encrypted_ratings_matrix:
        encrypted_row_result = []
        for row2 in encrypted_ratings_matrix:
            dot_product = dot_product_encrypted(evaluator, row1, row2, relin_key)
            encrypted_row_result.append(dot_product)
        encrypted_result.append(encrypted_row_result)

    # Decrypt and decode the result
    decrypted_result = []
    for row in encrypted_result:
        decrypted_row = [keys['decryptor'].decrypt(c) for c in row]
        decoded_row = [keys['encoder'].decode(c) for c in decrypted_row]
        decrypted_result.append(decoded_row)

    print("User-Item Ratings Matrix:\n", user_item_ratings)
    print("Plaintext Matrix Multiplication Result:\n", plaintext_result)
    print("Encrypted Matrix Multiplication Result (Decrypted for comparison):\n", decrypted_result)

if __name__ == "__main__":
    main()
