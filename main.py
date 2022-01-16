import argparse
import os
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    assert os.path.isdir(args.path), "Given path is not a directory"
    files = glob.glob(glob.escape(args.path) + "/*.mp4")
    print(files)
    for file in files:
        pass
