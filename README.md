# Homomorphic Encryption for Data Privacy

This project explores the application of homomorphic encryption to safeguard data privacy across various sectors, including healthcare, banking, and defense. Homomorphic encryption enables computations on encrypted data, allowing third-party services to perform operations without ever decrypting the sensitive information. This ensures that privacy is maintained while still allowing useful data analysis and decision-making.

Project Report: https://drive.google.com/file/d/1kMG8MH66vMHBvO_I6VoYuMctvVwq7vZ2/view?usp=sharing

## Overview

The project demonstrates a proof of concept using homomorphic encryption for matrix multiplication. Matrix multiplication is a core operation used in various models and computations, making it an ideal foundation for implementing and showcasing the potential of fully homomorphic encryption (FHE) in secure computation.

## Key Features

- **Fully Homomorphic Encryption (FHE)**: This project leverages FHE to allow computations on encrypted data, ensuring data privacy and security.
- **Applications**: This technology is useful in sectors such as healthcare, banking, elections, taxes, and military operations.
- **Proof of Concept**: Matrix multiplication implementation as a demonstration of the core functionality of homomorphic encryption.
- **Privacy**: Ensures sensitive data, such as medical records, financial information, and voting data, remain encrypted and secure while computations are performed.

## Applications

Homomorphic encryption has a wide range of applications, including but not limited to:

1. **Healthcare**: Enables secure processing of patient data, ensuring privacy while still facilitating valuable research and diagnosis.
2. **Banking**: Allows financial institutions to detect fraud and analyze spending patterns without compromising customer privacy.
3. **Elections**: Safeguards voter anonymity by encrypting ballots and ensuring that election results can be tallied without revealing private information.
4. **Military**: Protects sensitive operational data and communications from adversaries, ensuring the security and confidentiality of military operations.

## Libraries

The project utilizes the **OpenFHE** library (formerly known as PALISADE), which is a fully homomorphic encryption implementation designed for performance and scalability. OpenFHE supports various encryption schemes and is highly optimized for both research and practical applications.

## Installation

### Prerequisites

- C++ 14 or higher
- Git
- OpenFHE Library (OpenFHE supports several systems and can be built from source or installed via package managers)

### Installation Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/homomorphic-encryption-data-privacy.git
    cd homomorphic-encryption-data-privacy
    ```

2. **Install OpenFHE library:**

    Follow the [OpenFHE installation guide](https://github.com/OpenFHEorg/openfhe-development) to set up the library on your system.

3. **Build the project:**

    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

4. **Run the proof of concept:**

    ```bash
    ./matrix_multiplication
    ```

## Usage

1. **Matrix Multiplication:**
   - The matrix multiplication implementation demonstrates the core functionality of homomorphic encryption.
   - Modify the input matrices to experiment with different data sets.

2. **Encrypting Data:**
   - Homomorphic encryption is applied to the input matrices before any computations are performed.
   - The resulting encrypted matrix is then decrypted to reveal the correct result.

## Contribution

Contributions are welcome! Feel free to fork the repository, create a branch, and submit a pull request with your changes. If you find any issues or have suggestions for improvements, please create an issue or open a discussion.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OpenFHE Library** for providing the foundational tools for homomorphic encryption.
- The research team who contributed to this paper and the implementation: Omar Loudghiri, Nick Harms, Kaia Kanj, Tom Hua, Kamsi Eneh, and Stephen Yen.

## Contact

For any questions or inquiries, feel free to contact the project maintainers:

- Omar Loudghiri: [oxl51@case.edu](mailto:oxl51@case.edu)
- Nick Harms: [nmh84@case.edu](mailto:nmh84@case.edu)
- Kaia Kanj: [kmk233@case.edu](mailto:kmk233@case.edu)
- Tom Hua: [yxh1165@case.edu](mailto:yxh1165@case.edu)
- Kamsi Eneh: [kre40@case.edu](mailto:kre40@case.edu)
- Stephen Yen: [sxy747@case.edu](mailto:sxy747@case.edu)



