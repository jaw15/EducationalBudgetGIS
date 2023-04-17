# imports
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# global vars
budgets = "C:\\Users\\2020c\\Documents\\GIS\\PSProject\\PSProject\\grouped_outcomes.csv"
edu_outcomes = "C:\\Users\\2020c\\Documents\\GIS\\PSProject\\FeatureRegularization\\education_outcomes.csv"


# X = []
# y = []


def feature_selection():
    print("setting up data...")
    bud_dict = dict()
    edu_dict = dict()

    print("parsing tables...")
    with open(budgets, 'r') as bud:
        with open(edu_outcomes, 'r') as edu:
            for b_str in bud.readlines():
                # print(b_str)
                b_str = b_str[:-1] if b_str.endswith('\n') else b_str
                b_row = b_str.split(',')
                if b_row[0] == "'CONUM'":
                    # print("b title field")
                    continue
                # print(b_row)
                b_loc = b_row[0]
                # print(b_loc)
                bud_dict[b_loc] = b_row
            for e_str in edu.readlines():
                e_str = e_str[:-1] if e_str.endswith('\n') else e_str
                e_row = e_str.split(',')
                if e_row[0] == "GEOID_TXT":
                    # print("e title field")
                    continue
                e_loc = e_row[0]
                edu_dict[e_loc] = e_row

    # print(bud_dict)
    # print(edu_dict)

    print("bringing data into numpy structures...")
    shared = set(bud_dict.keys()).intersection(set(edu_dict.keys()))
    # print(len(bud_dict), len(edu_dict), len(shared))

    X = np.array([np.array([int(val) for val in bud_dict[shared_cty][1:]]) for shared_cty in shared])
    y = np.array([edu_dict[shared_cty][0] for shared_cty in shared])
    print(X.shape)

    print("standardizing features...")
    # standardize features
    scaler = StandardScaler()
    print(X.shape)
    scaler.fit(X)

    print("identifying best features...")
    sel_ = SelectFromModel(
        LogisticRegression(C=0.5, penalty='l1', solver='liblinear', random_state=10), threshold=5)
    sel_.fit(scaler.transform(X), y)

    feat_bools = sel_.get_support()
    print(sel_.threshold_)
    X_selected = sel_.transform(scaler.transform(X))
    # print(X_selected)


    rel_fields = []
    with open(budgets, 'r') as buds:
        fields = buds.readline()[:-1].split(',')[1:]
        for i in range(len(fields)):
            if feat_bools[i]:
                print('relevant:', fields[i])
                rel_fields.append(fields[i])
            else:
                print('irrelevant:', fields[i])

        print("relevant fields:", rel_fields)


feature_selection()
