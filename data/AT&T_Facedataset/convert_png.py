import os
from pathlib import Path
from PIL import Image

def foo():
    paths = list(Path(".").rglob("*pgm"))
    for file in paths:
        filename, extension  = os.path.splitext(file)
        if extension == ".pgm":
            new_file = f"{filename}.png"
            with Image.open(file) as im:
                im.save(new_file)

foo()