import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
datasetPath = '/home/4t/SG/BRACS_ROI'
sys.path.append(rootPath)

"""
datasets selection
0: bird-40
"""

global_data_selection = 12
global_deep_learning_selection = 12

class Configuration(object):
    def __init__(self, data_selection, deep_learning_selection):
        self.data_selection = data_selection
        self.deep_learning_selection = deep_learning_selection

        # setting risk dataset
        # 2020.07.11 remove the datasets file to parent directory
        self.data_dict = {
            0: 'cars_9110',
            1: 'cars_9113',
            2: 'cars_9120',
            3: 'cars_9140',
            4: 'fgvc_915', 
            5: 'fgvc_18213',
            6: 'fgvc_9110',
            7: 'fgvc_9120',
            8: 'cub_9110',
            9: 'cub_9113',
            10: 'cub_9120',
            11: 'cub_9140',
            13: 'transform_r50',
            14:'hosp_test_b32_e64_lr0005_new',
            15:'transform_r101_d169',
            16:'CCT',
            17:'hosp_r50',
            12:'/home/ssd0/SG/sheeraz/result_archive/risk_elem/Bracs_/',
        }
        self.class_num_dict = {
            0: 196,
            1: 196,
            2: 196,
            3: 196,
            4: 100,
            5: 100,
            6: 100,
            7: 100,
            8: 200,
            9: 200,
            10: 200,
            11: 200,
            12: 7,
        }

        # image_dataset: image dataset divide
        # risk_dataset: the dataset for risk model, which includes all_data_info.csv, train.csv,
        #             val.csv, test.csv, decision_tree_rules_clean.txt, decision_tree_rules_info
        # npy_dataset: the dataset from interpretable neural network
        # dataset2csv: extract the risk csv of each class from the npy_dataset, which uses to generate the rules of each class
        # dataset2mulcsv: extract all risk csv of all class from the bpy_dataset, which uses to merge the risk_dataset
        # rules: save the rules of each class, which uses to merge all the rules for risk dataset
        # base_risk_nums: how many prototypes are there for each class

        self.data_path = self.data_dict[self.data_selection]
        self.image_dataset_path = os.path.join(self.data_path, 'image_dataset')
        self.risk_dataset_path = os.path.join(self.data_path, 'risk_dataset')
        self.npy_dataset_path = os.path.join(self.data_path, 'npy_dataset')
        self.data2csv_path = os.path.join(self.data_path, 'data2csv')
        self.data2mulcsv_path = os.path.join(self.data_path, 'data2mulcsv')
        self.rules_path = os.path.join(self.data_path, 'rules')
        self.base_risk_nums = 10
        self.base_risk_list = ['101', '50', 'Wide50']


        # setting epochs
        # these parameters are not used now
        self.train_size = 20
        self.deep_learning_epochs = 1
        self.risk_epochs = 100

        # setting risker
        # 2020.7.11 change minimum_observation_num to percentage
        #
        # minimum_observation_num: the rule matchs minimun of data
        # rule_acc: the lowest accuracy of this rule on the validation

        self.interval_number_4_continuous_value = 50
        # self.learing_rate = 0.001    default

        self.learing_rate = 0.0005
        self.risk_training_epochs = 50
        self.learn_variance = True
        self.apply_function_to_weight_classifier_output = True
        self.minimum_observation_num = 0.0
        self.rule_acc = 0.0
        self.risk_confidence = 0.90
        self.model_save_path = os.path.join(self.risk_dataset_path, 'tf_model')


        # setting decision_tree
        self.match_gini = 0.2
        self.unmatch_gini = 0.0001
        self.tree_depth = 1
        self.generate_rules = False

        self.raw_data_path = None
        self.raw_decision_tree_rules_path = None
        self.decision_tree_rules_path = os.path.join(self.risk_dataset_path, 'decision_tree_rules_clean.txt')
        self.info_decision_tree_rules_path = os.path.join(self.risk_dataset_path, 'decision_tree_rules_info.txt')
        self.train = None

        # the frame is used
        self.risk_model_type = 'f'  # torch or tf

    def get_parent_path(self):
        return self.risk_dataset_path

    def get_npy_dataset_path(self):
        return self.npy_dataset_path

    def get_data2csv_path(self):
        return self.data2csv_path

    def get_data2mulcsv_path(self):
        return self.data2mulcsv_path

    def get_risk_dataset_path(self):
        return self.risk_dataset_path

    def get_class_num(self):
        return self.class_num_dict[self.data_selection]

    def get_rules_dataset_path(self):
        return self.rules_path

    def get_raw_decision_tree_rules_path(self):
        return self.raw_decision_tree_rules_path

    def get_info_decision_tree_rules_path(self):
        return self.info_decision_tree_rules_path

    def get_decision_tree_rules_path(self):
        return self.decision_tree_rules_path

    def get_raw_data_path(self):
        return self.raw_data_path

    def get_train_path(self):
        return self.train

    def get_all_data_path(self):
        return os.path.join(self.risk_dataset_path, 'all_data_info.csv')
