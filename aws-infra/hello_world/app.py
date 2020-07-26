import json
import pandas as pd 
import numpy as np
import xml.etree.ElementTree as et
import boto3

# import requests
def parse(myroot : et) -> [list, list, list, list, list, list, list, list, list, list, list, list, list, list, list, list, int]:
    
    ids = []
    first_names = []
    last_names = []
    ssn = []
    address = []
    score = []


    types = []
    last_activity = []
    high_credit = []
    balance = []
    active = []
    status = []
    overdue = []
    overdue_count = []

    month_payment = []
    status_payment = []
    
    payments_per_tradeline = []
    
    for info in myroot:


        if info.tag == 'ID':
            ids.append(info.text)
        if info.tag == 'name':
            for name in info:
                if name.tag == 'first_name':
                    first_name = name.text
                    first_names.append(name.text)
                if name.tag == 'last_name':
                    last_name = name.text
                    last_names.append(name.text)
        if info.tag == 'SSN':
            ssn.append(info.text)
        if info.tag == 'address':
            address.append(info.text)
        if info.tag == 'score':
            score.append(info.text)
        if info.tag == 'tradelines':
            tradelines = info.findall('tradeline')
            count = 0 # need to -1
#             a.append(len(tradelines))
            for tradeline in tradelines:
                count +=1
                for line in tradeline:

                    if line.tag == 'type':
                        types.append(line.text)
                    if line.tag == 'last_activity':
                        last_activity.append(line.text)
                    if line.tag == 'high_credit':
                        high_credit.append(line.text)
                    if line.tag == 'balance':
                        balance.append(line.text)
                    if line.tag == 'active':
                        active.append(line.text)
                    if line.tag == 'status':
                        status.append(line.text)



                    if line.tag == 'overdue':
                        overdue.append(line.text)
                        overdue_count.append(count)

                    if line.tag == 'payments':
                        payments = line.findall('payment')
                        payments_per_tradeline.append(len(payments))
                        for payment in payments:
                            for pay in payment:
                                if pay.tag == 'month':
                                    month_payment.append(pay.text)
                                if pay.tag == 'status':
                                    status_payment.append(pay.text)
    return [ids, first_names, last_names, ssn, address, types, last_activity, high_credit, balance, active, status, overdue, month_payment, status_payment, overdue_count, payments_per_tradeline]



def outer(l : list, total: list) -> list:
    elem = l[-1]
    return [elem for x in range(sum(total))]

def tradeline_outer(tradeline_attr: list, total_per_tradeline: list) -> list:
    ret = []
    count = 0
    for tradeline_len in total_per_tradeline:
        elems = [tradeline_attr[count] for x in range(tradeline_len)]
        count +=1
        ret += elems
    return ret
def overdue_col(overdue : list, overdue_count : list, total_per_tradeline: list) -> list:
    if overdue == None:
        return [0 for x in sum(total_per_tradeline)]
    ret = []
    overdue_count = [x - 1 for x in overdue_count]
    count = 0
    count_overdue = 0
    for tradeline_len in total_per_tradeline:
        if count in overdue_count:
            elems = [overdue[count_overdue] for x in range(tradeline_len)]
            count +=1
            count_overdue +=1
            ret += elems
        else:
            elems = [0 for x in range(tradeline_len)]
            ret += elems
            count +=1
    return ret

def trade_line_flag(payments_per_tradeline: list) -> list:
    ret = []
    count = 0
    for payment in payments_per_tradeline:
        elem = [count for x in range(payment)]
        count +=1
        ret.extend(elem)
    return ret

def create_cols(tup : list) -> list:
    
    payments_per_tradeline = tup[-1]
    
    
    tup[0] = outer(tup[0], payments_per_tradeline)
    tup[1] = outer(tup[1], payments_per_tradeline)
    tup[2] = outer(tup[2], payments_per_tradeline)
    tup[3] = outer(tup[3], payments_per_tradeline)
    tup[4] = outer(tup[4], payments_per_tradeline)
    
    tup[5] = tradeline_outer(tup[5], payments_per_tradeline)
    

    # 6 - 11
    tup[6] = tradeline_outer(tup[6], payments_per_tradeline)
    tup[7] = tradeline_outer(tup[7], payments_per_tradeline)
    tup[8] = tradeline_outer(tup[8], payments_per_tradeline)
    tup[9] = tradeline_outer(tup[9], payments_per_tradeline)
    tup[10] = tradeline_outer(tup[10], payments_per_tradeline)
    
    tup[11] = overdue_col(tup[11], tup[14], payments_per_tradeline)

    
    tup.pop()
    tup.pop()

    trade_line_f = trade_line_flag(payments_per_tradeline)
    tup.append(trade_line_f)
    
        
    return tup



        
def create_dataframe(file: list) -> pd.core.frame.DataFrame:
    cols = ['ID', 'first_name', 'last_name', 'SSN', 'address', 
                'type', 'last_activity' , 'high_credit', 'balance', 'active', 'status',
                'overdue', 'month_payment', 'status_payment', 'tradeline_flag']
    cols_dict = {}
    index = 0
    for l in file:
        cols_dict[cols[index]] = l
        index +=1
    return pd.DataFrame(cols_dict)



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
def normalize_continuous(df, numerical_cols): # change to standardize
    for col in numerical_cols:
        df[col] = (df[col] - df[col].mean())/ df[col].std()
    return df

def convert_to_numeric(df):
    for col in df.columns.values:
        df[col] = pd.to_numeric(df[col], errors = 'ignore')
    return df

class Preprocess:
    def __init__(self, df):
        self.df = df
        self.columns = df.columns.values
    
    def __call__(self):
        df = self.df
        df.drop(columns={'ID', 'first_name', 'last_name', 'SSN', 'address', 'last_activity', 'month_payment', 'tradeline_flag'}, inplace = True)
        print('COLUMNS: ', df.columns.values)
        df = convert_to_numeric(df)
        # score - normal
        # high credit - continuous
        df = normalize_continuous(df, ['high_credit'])
        # balance -- log -- add 1 because of the number of 0's (45%)
#         df['balance'] = df['balance'].apply(lambda x: x + 1) # not too sure here
        df = normalize_continuous(df, ['balance'])
        df = normalize_continuous(df, ['overdue'])
        # active from true / false to 0 / 1
        df = feature_bool(df, 'active')
        
        # ohe_cols = ['type', 'status', 'status_payment']#, 'regions', 'divisions']
        # df = one_hot_encode(df, ohe_cols)

        types_vector = ['AUTO', 'EDU', 'REVOLVING']
        status_vector = ['COLLECTIONS', 'CURRENT', 'LATE', 'PAID']
        status_payment_vector = ['30 DAYS LATE', '60 DAYS LATE OR MORE', 'ON TIME']

        vector = []
        for index in range(df.shape[0]):
            vec_types = [1 if df['type'].iloc[index] == t else 0 for t in types_vector]
            vec_status = [1 if df['status'].iloc[index] == t else 0 for t in status_vector]
            vec_status_payment = [1 if df['status_payment'].iloc[index] == t else 0 for t in status_payment_vector]
            
            vec = vec_types + vec_status + vec_status_payment
            
            vector.append(vec)
    

        df_no_cat = df.drop(columns = {'type', 'status','status_payment'})

        arr = np.concatenate([df_no_cat.values, vector], axis=1)

        df = pd.DataFrame(arr)
        

        print(df.iloc[0])
        
        return df, np.array(df)
    
def lambda_handler(event, context):


    myroot = et.fromstring(event['body'])
    file_data = parse(myroot)
    file_data = create_cols(file_data)
    raw = create_dataframe(file_data)

    prepro_in = Preprocess(raw)
    df, X = prepro_in()
    
    payload = [[str(entry) for entry in row] for row in df.values]
    payload = '\n'.join([','.join(row) for row in payload])

    endpoint = 'xgboost-2020-07-02-20-54-46-762'
    runtime = boto3.Session().client('sagemaker-runtime')
    response = runtime.invoke_endpoint(EndpointName = endpoint,
                                        ContentType = 'text/csv',
                                        Body = payload)

    result = response['Body'].read().decode('utf-8')
    Y_pred = np.fromstring(result, sep = ',')


    res = sum(Y_pred) / len(Y_pred)
    # train score stats
    mean = 1108.95
    std = 351.19
    res = res * std + mean
    res = int(res)
    
    








    
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

    # try:
    #     ip = requests.get("http://checkip.amazonaws.com/")
    # except requests.RequestException as e:
    #     # Send some context about this error to Lambda Logs
    #     print(e)

    #     raise e
    return {
        "statusCode": 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        "body": str(res)
    }
