import sys
import os.path
from pathlib import Path

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
#
#  philipp@mango:~/facerec/data/at$ tree
#  .
#  |-- README
#  |-- s1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- s2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm

if len(sys.argv) != 3:
    print("usage: create_csv <base_path> <file_extension>")
    exit()

BASE_PATH = sys.argv[1]
SEPARATOR = ";"
EXSTENSION = sys.argv[2]
out_csv = "file_label.csv"

paths = list(Path(".").rglob(f"*{EXSTENSION}"))
full_paths = [p.resolve() for p in paths]

label = 0
index = 0
lines = []
for dirname, dirnames, filenames in os.walk(BASE_PATH):
    for subdirname in dirnames:
        subject_path = os.path.join(dirname, subdirname)
        for filename in os.listdir(subject_path):
            # abs_path = "%s/%s" % (subject_path, filename)
            # print(abs_path)
            if (os.path.splitext(filename)[1] == f".{EXSTENSION}"):
                lines.append(f"{full_paths[index]}{SEPARATOR}{label}\n")
                index += 1
        label += 1

with open(out_csv, mode="w", newline="") as out_fp:
    out_fp.writelines(lines)