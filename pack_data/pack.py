import tarfile

with tarfile.open("cats-dataset.tar.gz", "w:gz") as tar:
    tar.add("cats-dataset")
