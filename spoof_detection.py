import joblib
from pathlib import Path
import sys
import numpy as np


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage python spoof_detection.py [path_to_feature_vec].npz")
        exit(-1)

    clf = joblib.load(Path('models', 'spoof_model.pkl'))
    p_fv = Path(sys.argv[1])
    if p_fv.exists() is False:
        print("Feature vector file doesn't exist")
    npdata = np.load(p_fv)
    fv = npdata['arr_0']
    prediction = clf.predict_proba(fv)
    prob = prediction[0][1]
    sys.exit(int(prob * 100))