import colour
from colour_demosaicing import (
    ROOT_RESOURCES_EXAMPLES,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)
import csv
from scipy.interpolate import interp1d

def mosaic(image):
    return mosaicing_CFA_Bayer(image, pattern='RGGB')

def demosaic(image):
    return demosaicing_CFA_Bayer_bilinear(image, pattern='RGGB')

def gamma_expansion(image, gamma=2.2):
    return image ** (1.0 / gamma)

def gamma_compression(image, gamma=2.2):
    return image ** gamma

def get_crf(camera_idx):
    """ Obtains desired Camera Response Function (CRF)
        Inputs:
            camera_idx - index of CRF to obtain
        Outputs
            I - Scene Irradiance (1024, 1) array
            B - Image Brightness (1024, 1) array
        """
    # desired camera index
    camera_idx *= 6

    with open("vita/data/degradations/agllnet/dorfCurves.txt", 'r') as fobj:
        reader = csv.reader(fobj, delimiter=' ')
        rows = list(reader)

        I = rows[camera_idx + 3] 
        B = rows[camera_idx + 5]

        I = [float(val) for val in filter(
                lambda x: True if len(x) > 0 else False, I)]
        B = [float(val) for val in filter(
                lambda x: True if len(x) > 0 else False, B)]

        return I, B
    
def remove_dups(a):
    seen = set()
    uniq = []
    dups_idx = []
    for i, x in enumerate(a):
        if x not in seen:
            uniq.append(x)
            seen.add(x)
        else:
            dups_idx.append(i)

    return uniq, dups_idx

def get_crf_icrf(camera_idx):
    I, B = get_crf(camera_idx)
    
    B_uniq, dups_idx = remove_dups(B)
    I_uniq = [I[i] for i in range(len(I)) if i not in dups_idx]

    crf = interp1d(I_uniq, B_uniq, kind='cubic')
    icrf = interp1d(B_uniq, I_uniq, kind='cubic')

    return crf, icrf