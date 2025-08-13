import os
import shutil
from os.path import join

import numpy as np
import pandas as pd
from get_distance import get_distance
from get_knn_count import get_knn_count


def my_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_risk_info(
    data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list, note
):
    # Get risk elem
    for layer in layers:

        for cnn in cnns:
            print("=== geting risk_info of {}_{} ===".format(cnn, layer))

            elem_name = elem_name_str.format(cnn)
            csv_dir = csv_dir_str.format(data_dir, cnn)
            num_class = (
                int(
                    pd.read_csv(join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None)
                    .to_numpy()
                    .flatten()[-1]
                )
                + 1
            )
            
            get_distance(layer, elem_name, num_class, csv_dir, "cosine", data_sets)
            get_knn_count(
                k_list, layer, elem_name, num_class, csv_dir, "cosine", data_sets
            )

    # Merge csv
    csv_path_list = []
    for cnn in cnns:
        for layer in layers:
            for elem in elems:
                if elem=='fangcha':
                    if layer=='x4':
                        continue
                    if cnn=='CCT':
                        continue
                if elem =='xs8' or elem=='xs1'or elem =='xs3' or elem=='xs5':
                        if layer == 'x4':
                            continue
                        if cnn == 'CCT':
                            continue
                if elem == 'paddingdis':
                    if cnn == 'CCT':
                        continue
                if elem == 'padknn8' or elem == 'padknn1':
                    if cnn == 'CCT':
                        continue
                if elem == 'xsdis':
                    if cnn == 'CCT':
                        continue
                    if layer == 'x4':
                        continue
                if elem == 'all3' or elem == 'all5':
                    if cnn == 'CCT':
                        continue
                    if layer == 'x4':
                        continue
                csv_path_list.append(
                        join(
                            csv_dir_str.format(data_dir, cnn),
                            "{}_{}_{}.csv".format(cnn, layer, elem),
                        )
                    )



    all_info = pd.read_csv(csv_path_list[0], header=None).to_numpy()[:, :2]

    for csv_path in csv_path_list:
        csv = pd.read_csv(csv_path, header=None).to_numpy()[:, 2:]
        print(csv_path)
        all_info = np.hstack((all_info, csv))

    pd.DataFrame(all_info).to_csv(
        "/home/ssd0/SG/sheeraz/result_archive/risk_elem/{}/risk_dataset{}/all_data_info.csv".format( #change path to the save distribution
            data_dir, note
        ),
        header=None,
        index=None,
    )

    all_info = all_info.tolist()

    for data_set in data_sets:
        #print(data_set)
        temp_csv = [all_info[0]]
        #print(temp_csv)
        for line in all_info[1:]:
            #print(line[0])
            #print(line[0].split("/")[1])
            if line[0].split("/")[1] == data_set:
                temp_csv.append(line)

        pd.DataFrame(temp_csv).to_csv(
            "/home/ssd0/SG/sheeraz/result_archive/risk_elem/{}/risk_dataset{}/{}.csv".format( #change path to the save distribution
                data_dir, note, data_set
            ),
            header=None,
            index=None,
        )


# get risk elem
if __name__ == "__main__":

    get_risk_info()
