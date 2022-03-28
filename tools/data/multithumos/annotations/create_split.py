from glob import glob
import os

SOURCE_FOLDER = 'backup/*.txt'

VAL_DIR = 'val'
TEST_DIR = 'test'


def write_out(filename, content, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    root, ext = os.path.splitext(filename)
    filename = root + '_' + directory + ext
    full_path = os.path.join(directory, filename)

    with open(full_path, 'w') as f:
        f.write('\n'.join(content))

    return full_path

def separate(lines):
    val_lines = []
    test_lines = []

    for line in lines:
        if 'test' in line:
            test_lines.append(line)
        elif 'validation' in line:
            val_lines.append(line)

    return val_lines, test_lines

for file in glob(SOURCE_FOLDER):
    with open(file, 'r') as f:
        lines = [l.strip('\n') for l in f]

    val_lines, test_lines = separate(lines)

    write_out(os.path.basename(file), val_lines, VAL_DIR)
    write_out(os.path.basename(file), test_lines, TEST_DIR)
