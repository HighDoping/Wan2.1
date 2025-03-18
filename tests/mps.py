import torch


# test matmul for different devices
def test_matmul_same_dtypes():
    # List of dtypes to test
    dtypes = [
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.complex32,
        torch.complex64,
        torch.complex128,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.quint8,
        torch.qint8,
        torch.qint32,
        torch.quint4x2,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]

    # List of available devices
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")

    # Create test matrices
    matrix_size = (128, 128)

    print("Matrix Multiplication Same Dtype Support Test")
    print("-" * 70)
    print(f"{'Dtype':<30} {'Device':<10} {'Status':<20}")
    print("-" * 70)

    # Test each dtype and device combination
    for dtype in dtypes:
        for device in devices:
            try:
                # Create test matrices with same dtype on specified device
                a = torch.randn(*matrix_size).to(dtype).to(device)
                b = torch.randn(*matrix_size).to(dtype).to(device)

                # Try matrix multiplication
                _=torch.matmul(a, b)
                status = "Supported ✓"

            except (RuntimeError, TypeError) as e:
                status = "Not Supported ✗"

            print(f"{str(dtype):<30} {device:<10} {status:<20}")
        print("-" * 70)


if __name__ == "__main__":
    test_matmul_same_dtypes()
