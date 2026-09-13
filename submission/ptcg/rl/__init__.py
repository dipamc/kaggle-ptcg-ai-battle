import os
import sys

# cg (competition SDK) lives in <repo>/data; loads libcg per-platform.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
