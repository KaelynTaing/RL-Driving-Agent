import torch, numpy, sklearn, sys

print("NumPy:", numpy.__file__)
print("PyTorch:", torch.__file__)
print("scikit-learn:", sklearn.__file__)

import ctypes, glob

print("\nLoaded libomp libraries:")
for lib in glob.glob("/**/*libomp*.dylib", recursive=True):
    print(lib)
