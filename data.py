import GEOparse
import numpy as np


num_features = 22283


def parse_file(path, l_C, r_C, l_RA, r_RA, num_samples):

    gds = GEOparse.get_GEO(filepath=path)
    assert num_features == int(gds.metadata['feature_count'][0]),\
           "Found different number of features!"

    # Prepare container for data
    Xs = np.empty(shape=(num_samples, num_features), dtype=float)
    ys = np.empty(shape=(num_samples, ), dtype=int)
    curr_sample = 0

    # Get gene expression data from columns representing RA patients/controls
    cols = gds.table.columns
    for key in cols:
        if 'GSM' in key:
            if l_C <= key[-2:] and key[-2:] <= r_C:
                # Class = control person
                ys[curr_sample] = 0
            elif l_RA <= key[-2:] and key[-2:] <= r_RA:
                # Class = RA patient
                ys[curr_sample] = 1
            else:
                # Other data - not interested
                continue

            vals = np.zeros(shape=(num_features,), dtype=float)
            for i in range(0, len(gds.table[key])):
                if type(gds.table[key][i]) == str:
                    try:
                        vals[i] = float(gds.table[key][i])
                    except ValueError:
                        vals[i] = float(0)
                else:
                    vals[i] = gds.table[key][i]

            Xs[curr_sample, :] = vals
            if l_C <= key[-2:] and key[-2:] <= r_C:
                ys[curr_sample] = 0
            elif l_RA <= key[-2:] and key[-2:] <= r_RA:
                ys[curr_sample] = 1
            curr_sample += 1

    return (Xs, ys)


def get_RA_data():

    (Xs1, ys1) = parse_file("./GDS5401_full.soft",
                            l_C='01', r_C='10', l_RA='21', r_RA='30',
                            num_samples=20)
    (Xs2, ys2) = parse_file("./GDS5402_full.soft",
                            l_C='00', r_C='00', l_RA='18', r_RA='27',
                            num_samples=10)
    (Xs3, ys3) = parse_file("./GDS5403_full.soft",
                            l_C='04', r_C='13', l_RA='14', r_RA='26',
                            num_samples=23)

    Xs = np.concatenate((Xs1, Xs2, Xs3), axis=0)
    ys = np.concatenate((ys1, ys2, ys3), axis=0)
    print Xs.shape, ys.shape

    return (Xs, ys)


if __name__=='__main__':
    get_RA_data()
