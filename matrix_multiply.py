import numpy as np
import matplotlib.pyplot as plt
import time
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

def matrix_multiply(matrix1, matrix2, keys, block_size, scale):
    """Multiply two matrices in an encrypted manner."""
    encoder, evaluator, decryptor = keys['encoder'], keys['evaluator'], keys['decryptor']
    encryptor = keys['encryptor']  # Define encryptor here
    n = matrix1.shape[0]
    result_matrix = np.zeros((n, n), dtype=complex)

    for col_idx in range(matrix2.shape[1]):
        vector = matrix2[:, col_idx]
        encrypted_vector = encrypt_vector(encoder, encryptor, vector, scale, block_size)
        
        result_column = np.zeros(n, dtype=complex)
        for i in range(n // block_size):
            for j in range(n // block_size):
                block = matrix1[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                rotation_keys = {k: keys['keygen'].generate_rot_key(k) for k in range(len(block))}
                product = evaluator.multiply_matrix(encrypted_vector[j], block, rotation_keys, encoder)
                if j == 0:
                    aggregated_product = product
                else:
                    aggregated_product = evaluator.add(aggregated_product, product)
            
            decrypted_block = decryptor.decrypt(aggregated_product)
            decoded_block = encoder.decode(decrypted_block)
            result_column[i * block_size:(i + 1) * block_size] = decoded_block

        result_matrix[:, col_idx] = result_column

    return result_matrix

def dot_product_encrypted(evaluator, encrypted_vector1, encrypted_vector2, relin_key):

    if len(encrypted_vector1) != len(encrypted_vector2):
        raise ValueError("Encrypted vectors must have the same length.")

    # Initialize an encrypted zero if available, or use the first product
    encrypted_sum = None

    # Iterate through elements of vectors, multiply element-wise, and add to the sum
    for encrypted_val1, encrypted_val2 in zip(encrypted_vector1, encrypted_vector2):
        #print("here")
        encrypted_product = evaluator.multiply(encrypted_val1, encrypted_val2, relin_key)
        if encrypted_sum is None:
            encrypted_sum = encrypted_product
        else:
            encrypted_sum = evaluator.add(encrypted_sum, encrypted_product)

    return encrypted_sum

def mat_mult_performance_evaluator(MATRIX_SIZE, BLOCK_SIZE,SCALING_FACTOR):


    times_fhe = []
    times_np = []
    errors = []

    for _ in range(10):
        matrix1 = create_complex_matrix(MATRIX_SIZE)
        matrix2 = create_complex_matrix(MATRIX_SIZE)

        start_time_fhe = time.time()
        result = matrix_multiply(matrix1, matrix2, keys, BLOCK_SIZE, SCALING_FACTOR)
        end_time_fhe = time.time()

        start_time_np = time.time()
        result_np = np.matmul(matrix1, matrix2)
        end_time_np = time.time()

        error_np = np.subtract(result_np, result)
        norm_error = np.linalg.norm(error_np)

        times_fhe.append(end_time_fhe - start_time_fhe)
        times_np.append(end_time_np - start_time_np)
        errors.append(norm_error)
        
    # Calculate statistical measures
    mean_fhe = np.mean(times_fhe)
    std_fhe = np.std(times_fhe)
    median_fhe = np.median(times_fhe)
    range_fhe = np.ptp(times_fhe)

    mean_np = np.mean(times_np)
    std_np = np.std(times_np)
    median_np = np.median(times_np)
    range_np = np.ptp(times_np)

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    range_error = np.ptp(errors)

    print(f"Matrix size: {MATRIX_SIZE} Block size: {BLOCK_SIZE}")
    print(f"Encrypted Matrix Multiplication Time - Mean: {mean_fhe}, Std: {std_fhe}, Median: {median_fhe}, Range: {range_fhe}")
    print(f"Normal Matrix Multiplication Time - Mean: {mean_np}, Std: {std_np}, Median: {median_np}, Range: {range_np}")
    print(f"Error Norm - Mean: {mean_error}, Std: {std_error}, Median: {median_error}, Range: {range_error}")
        
        
    return mean_fhe, mean_np, mean_error

# Main parameters
MATRIX_SIZE = 2
BLOCK_SIZE = 2
SCALING_FACTOR = 1 << 30  # Define scaling factor directly

#print("Pre CKKS setup")

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

#print("Post CKKS setup")

# Generating matrices and performing encrypted multiplication
matrix1 = create_complex_matrix(MATRIX_SIZE)
matrix2 = create_complex_matrix(MATRIX_SIZE)

#mean_fhe, mean_np, mean_error = mat_mult_performance_evaluator(MATRIX_SIZE, BLOCK_SIZE, SCALING_FACTOR)
#print(matrix1)

matrix_sizes = [2, 4, 8, 16]
block_sizes = [2, 2, 2, 2]
mean_fhe_times = []
mean_np_times = []
mean_errors = []

for matrix_size, block_size in zip(matrix_sizes, block_sizes):
    mean_fhe, mean_np, mean_error = mat_mult_performance_evaluator(matrix_size, block_size, SCALING_FACTOR)
    mean_fhe_times.append(mean_fhe)
    mean_np_times.append(mean_np)
    mean_errors.append(mean_error)

# Plotting
plt.figure(figsize=(10, 7))
plt.subplot(311)
plt.plot(matrix_sizes, mean_fhe_times, marker='o')
plt.title('Mean Encrypted Multiplication Times')
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')

plt.subplot(312)
plt.plot(matrix_sizes, mean_np_times, marker='o', color='green')
plt.title('Mean Normal Multiplication Times')
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')

plt.subplot(313)
plt.plot(matrix_sizes, mean_errors, marker='o', color='red')
plt.title('Mean Error Norms')
plt.xlabel('Matrix Size')
plt.ylabel('Error Norm')

plt.tight_layout()
plt.show()

result = matrix_multiply(matrix1, matrix2, keys, BLOCK_SIZE, SCALING_FACTOR)

'''
print("First matrix:\n", matrix1)
print("Second matrix:\n", matrix2)
print("Plaintext multiplication: \n", np.matmul(matrix1, matrix2))
print("Encrypted matrix multiplication result:\n", result)
'''

relin_key = keys['keygen'].relin_key

# Generate two random vectors
vector1 = np.random.rand(MATRIX_SIZE) + 1j * np.random.rand(MATRIX_SIZE)
vector2 = np.random.rand(MATRIX_SIZE) + 1j * np.random.rand(MATRIX_SIZE)

# Compute dot product in plaintext
plaintext_dot_product = np.dot(vector1, vector2)

# Encrypt vectors
encrypted_vector1 = encrypt_vector(keys['encoder'], keys['encryptor'], vector1, SCALING_FACTOR, BLOCK_SIZE)
encrypted_vector2 = encrypt_vector(keys['encoder'], keys['encryptor'], vector2, SCALING_FACTOR, BLOCK_SIZE)

# Compute dot product in encrypted form
encrypted_dot_product = dot_product_encrypted(keys['evaluator'], encrypted_vector1, encrypted_vector2, relin_key)

# Decrypt the result of the encrypted dot product
decrypted_dot_product = keys['decryptor'].decrypt(encrypted_dot_product)
decoded_dot_product = keys['encoder'].decode(decrypted_dot_product)

'''
print("Vector 1:\n", vector1)
print("Vector 2:\n", vector2)
print("Plaintext dot product result:\n", plaintext_dot_product)
# print("Encrypted dot product result (decoded):\n", decoded_dot_product)
print("Encrypted dot product result (decoded):\n", np.sum(decoded_dot_product))
'''


