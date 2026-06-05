import os
import urllib.request

urls = {
    "train-images-idx3-ubyte.gz": [
        "https://storage.googleapis.com/cvdfoundation/mnist/train-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "https://github.com/cvdfoundation/mnist/raw/master/train-images-idx3-ubyte.gz"
    ],
    "train-labels-idx1-ubyte.gz": [
        "https://storage.googleapis.com/cvdfoundation/mnist/train-labels-idx1-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "https://github.com/cvdfoundation/mnist/raw/master/train-labels-idx1-ubyte.gz"
    ],
    "t10k-images-idx3-ubyte.gz": [
        "https://storage.googleapis.com/cvdfoundation/mnist/t10k-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "https://github.com/cvdfoundation/mnist/raw/master/t10k-images-idx3-ubyte.gz"
    ],
    "t10k-labels-idx1-ubyte.gz": [
        "https://storage.googleapis.com/cvdfoundation/mnist/t10k-labels-idx1-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        "https://github.com/cvdfoundation/mnist/raw/master/t10k-labels-idx1-ubyte.gz"
    ]
}

dest_dir = "./data/MNIST/raw"
os.makedirs(dest_dir, exist_ok=True)

# User-Agent header standard override to avoid 403 Forbidden
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

for filename, mirrors in urls.items():
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"{filename} already exists. Skipping.")
        continue
    
    success = False
    for url in mirrors:
        print(f"Trying to download {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"Successfully downloaded {filename}!")
            success = True
            break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
    
    if not success:
        print(f"ERROR: Could not download {filename} from any mirror.")
        exit(1)

print("All MNIST files downloaded successfully!")
