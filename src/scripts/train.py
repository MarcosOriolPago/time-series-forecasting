import argparse
from src.utils.load import load_to_df

def main():
    data = load_to_df(file_path)

    print(data)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file',
        required=True,
        type=str
    )

    args = parser.parse_args()
    file_path = args.file

    return file_path

if __name__ == "__main__":
    file_path = parse_args()
    main()