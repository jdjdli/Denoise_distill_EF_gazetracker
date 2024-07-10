import argparse
import tarfile
import os

parser = argparse.ArgumentParser(description='Arguments for using the data pre processing')
parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'),
                    help='absolute path to eye_data/, by default assumes same parent dir as this script')
opt = parser.parse_args()

data = os.path.join(os.getcwd(), 'data')


def un_gz(data_dir):
    lists = os.listdir(data_dir)
    for file_name in lists:
        t_file = tarfile.open(os.path.join(data_dir, file_name))
        t_file.extractall(path=data)


def main():
    un_gz(opt.data_dir)


if __name__ == '__main__':
    main()
