import pandas as pd
import time
import os
import numpy as np
import matplotlib.pyplot as plt


def convert_to_numeric(df):
    for col in df.columns.values:
        df[col] = pd.to_numeric(df[col], errors = 'ignore')
    return df

def to_datetime(df, col):
    df[col] = pd.to_datetime(df[col], format = '%Y')
    return df



def add_3_features(raw_data):
    
    
    frames = []
    a = time.time()
    for ids in raw_data.ID.unique(): #∆ from raw_ordered -> raw_data 
        x = 0
        y = 0
        z = 0
        seg = raw_data[raw_data['ID'] == ids]
        avg_high_credit = seg['high_credit'].mean()
        lower_high_credit = seg['high_credit'].describe()['25%']
        upper_high_credit = seg['high_credit'].describe()['75%']
        avg_high_credit_list = [avg_high_credit for x in range(seg.shape[0])]
        lower_high_credit_list = [lower_high_credit for y in range(seg.shape[0])]
        upper_high_credit_list = [upper_high_credit for z in range(seg.shape[0])]
        seg['avg_high_credit'] = avg_high_credit_list
        seg['lower_high_credit'] = lower_high_credit_list
        seg['upper_high_credit'] = upper_high_credit_list
        frames.append(seg)
    b = time.time()
    ordered = pd.concat(frames)
    c = time.time()
    print(b - a)
    print(c - b)
    
    return ordered




def sort_by_yr_month_payment_tradeline_flag(df):
    frames = []
    for ids in df.ID.unique():
        seg = df[df['ID'] == ids]
        # this would be the addtional code:
        for flag in seg.tradeline_flag.unique():
            seg1 = seg[seg['tradeline_flag'] == flag]
            seg1 = seg1.sort_values(by = 'yr_month_payment')
#         seg = seg.sort_values(by='yr_month_payment')
            frames.append(seg1)
    df = pd.concat(frames)
    return df




def sort_by_yr_month_payment_only(df):
    frames = []
    for ids in df.ID.unique():
        seg = df[df['ID'] == ids]
        seg = seg.sort_values(by='yr_month_payment')
        frames.append(seg)
    df = pd.concat(frames)
    return df


def plot_tradelines(low, filename):
    fig_dir = '../figs'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    tradeline_flags = low.tradeline_flag.unique()

    for tradeline_flag in tradeline_flags:
        tradeline_flag_df = low[low['tradeline_flag'] == tradeline_flag]
        plt.plot(tradeline_flag_df['yr_month_payment'], tradeline_flag_df['high_credit'], label=tradeline_flag)

    plt.legend()
    plt.xlabel('yr_month_payment')
    plt.xticks(rotation=90)
    plt.ylabel('high_credit') 
    plt.title('high_credit Graph')
    plt.savefig(os.path.join(fig_dir, filename+ '_per_tradeline.pdf'))
    plt.show()




# 209 + 89 = 298


# code for pca / kmeans

def select_subset(df, num: int):
    return df.loc[np.random.choice(df.index, num, replace=False)]


def create_additional_features(df1):
    
    import time
    a = time.time()

    # status
    late_tradelines = []
    collections_tradelines = []

    # type
    edu_tradelines = []
    auto_tradelines = []
    revolving_tradelines = []

    # status + type
    late_edu_tradelines = []
    collections_edu_tradelines = []


    late_auto_tradelines = []
    collections_auto_tradelines = []

    late_revolving_tradelines = []
    collections_revolving_tradelines = []

    # status + active
    current_non_active_tradelines = []

    # tradeline
    tradeline_counts = []
    tradelines_overdue = []

    for ids in tqdm(df1.ID.unique()):
        seg = df1[df1['ID'] == ids]


        # status
        count_late_tradelines = 0
        count_collections_tradelines = 0

        # type
        count_edu_tradelines = 0
        count_auto_tradelines = 0
        count_revolving_tradelines = 0



        # status + type
        count_late_edu_tradelines = 0
        count_collections_edu_tradelines = 0


        count_late_auto_tradelines = 0
        count_collections_auto_tradelines = 0


        count_late_revolving_tradelines = 0
        count_collections_revolving_tradelines = 0

        # status + active
        count_current_non_active_tradelines = 0

        # tradeline
        count_tradeline_counts = 0
        count_tradelines_overdue = 0

        # num tradelines so far
        count = 0
        num = 0

        for index, row in seg.iterrows():

            tradeline_counts.append(row['tradeline_flag'])

            if row['overdue'] > 0:
                count +=1
                tradelines_overdue.append(count)
            else:
                tradelines_overdue.append(count)



        tradeline_counts = [x+1 for x in tradeline_counts]

        for tradeline in seg.tradeline_flag.unique():
            seg1 = seg[seg['tradeline_flag'] == tradeline]
            if seg1.iloc[0]['status'] == 'LATE':
                count_late_tradelines +=1
                late_tradelines.extend([count_late_tradelines for x in range(seg1.shape[0])])
            else:
                late_tradelines.extend([count_late_tradelines for x in range(seg1.shape[0])])
            if seg1.iloc[0]['status'] == 'COLLECTIONS':
                count_collections_tradelines +=1
                collections_tradelines.extend([count_collections_tradelines for x in range(seg1.shape[0])])
            else:
                collections_tradelines.extend([count_collections_tradelines for x in range(seg1.shape[0])])


            # type
            if seg1.iloc[0]['type'] == 'EDU':
                count_edu_tradelines +=1
                edu_tradelines.extend([count_edu_tradelines for x in range(seg1.shape[0])])
            else:
                edu_tradelines.extend([count_edu_tradelines for x in range(seg1.shape[0])])

            if seg1.iloc[0]['type'] == 'AUTO':
                count_auto_tradelines +=1
                auto_tradelines.extend([count_auto_tradelines for x in range(seg1.shape[0])])
            else:
                auto_tradelines.extend([count_auto_tradelines for x in range(seg1.shape[0])])

            if seg1.iloc[0]['type'] == 'REVOLVING':
                count_revolving_tradelines +=1
                revolving_tradelines.extend([count_revolving_tradelines for x in range(seg1.shape[0])])
            else:
                revolving_tradelines.extend([count_revolving_tradelines for x in range(seg1.shape[0])])



            # status + type
            if seg1.iloc[0]['status'] == 'LATE' and seg1.iloc[0]['type']  == 'EDU':
                count_late_edu_tradelines +=1
                late_edu_tradelines.extend([count_late_edu_tradelines for x in range(seg1.shape[0])])
            else:
                late_edu_tradelines.extend([count_late_edu_tradelines for x in range(seg1.shape[0])])


            if seg1.iloc[0]['status'] == 'COLLECTIONS' and seg1.iloc[0]['type']  == 'EDU':
                count_collections_edu_tradelines +=1
                collections_edu_tradelines.extend([count_collections_edu_tradelines for x in range(seg1.shape[0])])
            else:
                collections_edu_tradelines.extend([count_collections_edu_tradelines for x in range(seg1.shape[0])])


            if seg1.iloc[0]['status'] == 'LATE' and seg1.iloc[0]['type']  == 'AUTO':
                count_late_auto_tradelines +=1
                late_auto_tradelines.extend([count_late_auto_tradelines for x in range(seg1.shape[0])])
            else:
                late_auto_tradelines.extend([count_late_auto_tradelines for x in range(seg1.shape[0])])

            if seg1.iloc[0]['status'] == 'COLLECTIONS' and seg1.iloc[0]['type']  == 'AUTO':
                count_collections_auto_tradelines +=1
                collections_auto_tradelines.extend([count_collections_auto_tradelines for x in range(seg1.shape[0])])
            else:
                collections_auto_tradelines.extend([count_collections_auto_tradelines for x in range(seg1.shape[0])])

            if seg1.iloc[0]['status'] == 'LATE' and seg1.iloc[0]['type']  == 'REVOLVING':
                count_late_revolving_tradelines +=1
                late_revolving_tradelines.extend([count_late_revolving_tradelines for x in range(seg1.shape[0])])
            else:
                late_revolving_tradelines.extend([count_late_revolving_tradelines for x in range(seg1.shape[0])])

            if seg1.iloc[0]['status'] == 'COLLECTIONS' and seg1.iloc[0]['type']  == 'REVOLVING':
                count_collections_revolving_tradelines +=1
                collections_revolving_tradelines.extend([count_collections_revolving_tradelines for x in range(seg1.shape[0])])
            else:
                collections_revolving_tradelines.extend([count_collections_revolving_tradelines for x in range(seg1.shape[0])])


    # status + active
    #     current_non_active_tradelines = []
    #     count_current_non_active_tradelines = 0


            if seg1.iloc[0]['status'] == 'CURRENT' and seg1.iloc[0]['active'] == False:
                count_current_non_active_tradelines +=1
                current_non_active_tradelines.extend([count_current_non_active_tradelines for x in range(seg1.shape[0])])
            else:
                current_non_active_tradelines.extend([count_current_non_active_tradelines for x in range(seg1.shape[0])])
    balance_to_high_credit = []
#     balance_to_score = []
    for index, row in df1.iterrows():
        balance_to_high_credit.append(row['balance'] / row['high_credit'])
#         balance_to_score.append(row['balance'] / row['score'])
    
    b = time.time()

    print('time: {}'.format(b - a))               
    return late_tradelines, collections_tradelines, edu_tradelines, auto_tradelines, revolving_tradelines, late_edu_tradelines, collections_edu_tradelines, late_auto_tradelines,collections_auto_tradelines, late_revolving_tradelines, collections_revolving_tradelines, current_non_active_tradelines, tradeline_counts, tradelines_overdue, balance_to_high_credit 

        

def append_additional_features(df, features: tuple):
    late_tradelines, collections_tradelines, edu_tradelines, auto_tradelines, revolving_tradelines, late_edu_tradelines, collections_edu_tradelines, late_auto_tradelines,collections_auto_tradelines, late_revolving_tradelines, collections_revolving_tradelines, current_non_active_tradelines, tradeline_counts, tradelines_overdue, balance_to_high_credit = features
    df['tradeline_counts'] = tradeline_counts
    df['tradelines_overdue'] = tradelines_overdue

    #status
    df['late_tradelines'] = late_tradelines
    df['collections_tradelines'] = collections_tradelines
    #type
    df['edu_tradelines'] = edu_tradelines
    df['auto_tradeliens'] = auto_tradelines
    df['collections_edu_tradelines'] = collections_edu_tradelines
    #status + type
    df['late_edu_tradelines'] = late_edu_tradelines
    df['collections_edu_tradelines'] = collections_edu_tradelines

    df['late_auto_tradelines'] = late_auto_tradelines
    df['collections_auto_tradelines'] = collections_auto_tradelines


    df['late_revolving_tradelines'] = late_revolving_tradelines
    df['collections_revolving_tradelines'] = collections_revolving_tradelines

    # status + active
    df['current_non_active_tradelines'] = current_non_active_tradelines
    df['balance_to_high_credit'] = balance_to_high_credit
#     df['balance_to_score'] = balance_to_score
    return df



class RegionMapping:
    def __init__(self, df):
        self.usregions = {"AA": {"Region": "None", "Division": "None"}, "AE": {"Region": "None", "Division": "None"}, "AP": {"Region": "west", "Division": "pacific"}, "AK": {"Region": "west", "Division": "pacific"}, "AL": {"Region": "south", "Division": "east south central"}, "AR": {"Region": "south", "Division": "west south central"}, "AZ": {"Region": "west", "Division": "mountain"}, "CA": {"Region": "west", "Division": "pacific"}, "CO": {"Region": "west", "Division": "mountain"}, "CT": {"Region": "northeast", "Division": "new england"}, "DC": {"Region": "south", "Division": "south atlantic"}, "DE": {"Region": "south", "Division": "south atlantic"}, "FL": {"Region": "south", "Division": "south atlantic"}, "GA": {"Region": "south", "Division": "south atlantic"}, "HI": {"Region": "west", "Division": "pacific"}, "IA": {"Region": "midwest", "Division": "west north central"}, "ID": {"Region": "west", "Division": "mountain"}, "IL": {"Region": "midwest", "Division": "east north central"}, "IN": {"Region": "midwest", "Division": "east north central"}, "KS": {"Region": "midwest", "Division": "west north central"}, "KY": {"Region": "south", "Division": "east south central"}, "LA": {"Region": "south", "Division": "west south central"}, "MA": {"Region": "northeast", "Division": "new england"}, "MD": {"Region": "south", "Division": "south atlantic"}, "ME": {"Region": "northeast", "Division": "new england"}, "MI": {"Region": "midwest", "Division": "east north central"}, "MN": {"Region": "midwest", "Division": "west north central"}, "MO": {"Region": "midwest", "Division": "west north central"}, "MS": {"Region": "south", "Division": "east south central"}, "MT": {"Region": "west", "Division": "mountain"}, "NC": {"Region": "south", "Division": "south atlantic"}, "ND": {"Region": "midwest", "Division": "west north central"}, "NE": {"Region": "midwest", "Division": "west north central"}, "NH": {"Region": "northeast", "Division": "new england"}, "NJ": {"Region": "northeast", "Division": "middle atlantic"}, "NM": {"Region": "west", "Division": "mountain"}, "NV": {"Region": "west", "Division": "mountain"}, "NY": {"Region": "northeast", "Division": "middle atlantic"}, "OH": {"Region": "midwest", "Division": "east north central"}, "OK": {"Region": "south", "Division": "west south central"}, "OR": {"Region": "west", "Division": "pacific"}, "PA": {"Region": "northeast", "Division": "middle atlantic"}, "RI": {"Region": "northeast", "Division": "new england"}, "SC": {"Region": "south", "Division": "south atlantic"}, "SD": {"Region": "midwest", "Division": "west north central"}, "TN": {"Region": "south", "Division": "east south central"}, "TX": {"Region": "south", "Division": "west south central"}, "UT": {"Region": "west", "Division": "mountain"}, "VA": {"Region": "south", "Division": "south atlantic"}, "VT": {"Region": "northeast", "Division": "new england"}, "WA": {"Region": "west", "Division": "pacific"}, "WI": {"Region": "midwest", "Division": "east north central"}, "WV": {"Region": "south", "Division": "south atlantic"}, "WY": {"Region": "west", "Division": "mountain"}}
        self.inversestate = {"Alabama": "AL", "Alaska": "AK", "American Samoa": "AS", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District Of Columbia": "DC", "Federated States Of Micronesia": "FM", "Florida": "FL", "Georgia": "GA", "Guam": "GU", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Marshall Islands": "MH", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Northern Mariana Islands": "MP", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Palau": "PW", "Pennsylvania": "PA", "Puerto Rico": "PR", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virgin Islands": "VI", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
        self.df = df
    def __call__(self):
        usregions = self.usregions
        inversestate = self.inversestate
        df = self.df
        addresses = list(df['address'].apply(lambda x: x[len(x)- 8 : len(x) - 6]).values)
        
        
        regions_mapped = [usregions[x]['Region'] for x in addresses]
        divisions_mapped = [usregions[x]['Division'] for x in addresses]
        df['regions'] = regions_mapped
        df['divisions'] = divisions_mapped
#         df.drop(columns= {'address'}, inplace = True)
        
        return df
    
    
import numpy as np
def feature_bool(df, col):
    active = df[col]
    active = [1 if x == True else 0 for x in active ]
    df[col] = active
    return df
def one_hot_encode(df, ohe_cols):
    """
    one hot endcode dataset
    df: input pandas dataset
    return: pandas dataset with all numerical values
    """
    return pd.get_dummies(df, columns=ohe_cols)

def remove_null_cols(df, thresh=0.08):
    """ 
    remove columns where # nulls > thresh % 
    
    df: input pandas dataframe
    thresh: maximum allotted percent of null values 
    
    return pandas dataframe with potentially dropped columns
    """
    
    # look at this
    # df.dropna(thresh=int(df.shape[0] * .9), axis=1)
    pct_null = df.isnull().sum() / len(df)
    missing_features = pct_null[pct_null > thresh].index
    return df.drop(missing_features, axis=1)
def normalize_continuous(df, numerical_cols): # change to standardize
    for col in numerical_cols:
        df[col] = (df[col] - df[col].mean())/ df[col].std()
    return df
def select_numerical(df):
    return df.select_dtypes(exclude=['object']).columns

def select_categorical(df):
    return df.select_dtypes(include=['object']).columns

def transform_log(df, cols):
    for col in cols:
        df[col] = np.log(df[col])
    return df

class PreprocessBaseline:
    def __init__(self, df):
        self.df = df
        self.columns = df.columns.values
    
    def __call__(self):
        df = self.df
        # score - normal
        # high credit - continuous
        df = normalize_continuous(df, ['score', 'high_credit'])
        # balance -- log -- add 1 because of the number of 0's (45%)
#         df['balance'] = df['balance'].apply(lambda x: x + 1) # not too sure here
        df = normalize_continuous(df, ['balance'])
        df = normalize_continuous(df, ['overdue'])
        # active from true / false to 0 / 1
        df = feature_bool(df, 'active')
        
        ohe_cols = ['type', 'status', 'status_payment']#, 'regions', 'divisions']
        df = one_hot_encode(df, ohe_cols)
        
        return df, np.array(df)

class Preprocess:
    def __init__(self, df, flag):
        self.df = df
        self.columns = df.columns.values
        self.flag = flag
    
    def __call__(self):
        df = self.df
        flag = self.flag
        # score - normal
        # high credit - continuous
        if flag == 'train':
            df = normalize_continuous(df, ['score', 'high_credit', 'balance', 'overdue'])
        else:
            df = normalize_continuous(df, ['high_credit', 'balance', 'overdue'])
        # balance -- log -- add 1 because of the number of 0's (45%)
#         df['balance'] = df['balance'].apply(lambda x: x + 1) # not too sure here
#         df = transform_log(df, ['balance', 'overdue'])
        # active from true / false to 0 / 1
        df = feature_bool(df, 'active')
        
        region_div = RegionMapping(df)
        df= region_div()
        
        
        # takes ~35 min -- append new features
        features = create_additional_features(df)
        df = append_additional_features(df, features)
        
        data_dir = 'preprocess'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if flag == 'train':
            df.to_csv(os.path.join(data_dir, 'raw_with_features.csv'))
        else:
            df.to_csv(os.path.join(data_dir, 'raw_val_with_features.csv'))
        
        if flag == 'train':
            df.drop(columns={'ID', 'first_name', 'last_name', 'SSN', 'address',
                             'last_activity', 'month_payment', 'tradeline_flag'}, inplace = True)
        else:
            ids = list(df['ID'])
            df.drop(columns={'ID', 'first_name', 'last_name', 'SSN', 'address',
                             'last_activity', 'month_payment', 'tradeline_flag'}, inplace = True)
        ohe_cols = ['type', 'status', 'status_payment', 'regions', 'divisions']
        
        df = one_hot_encode(df, ohe_cols)
        
        if flag == 'train' : 
            return df, np.array(df)
        else:
            return df, np.array(df), ids
        
        
        
        





