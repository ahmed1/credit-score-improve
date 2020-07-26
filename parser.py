import xml.etree.ElementTree as et
import pandas as pd


def parse(myroot : et, flag : str) -> [list, list, list, list, list, list, list, list, list, list, list, list, list, list, list, list, int]:
    
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
    if flag == 'train':
        return [ids, first_names, last_names, ssn, address, score, types, last_activity, high_credit, balance, active, status, overdue, month_payment, status_payment, overdue_count, payments_per_tradeline]
    else:
        return [ids, first_names, last_names, ssn, address, types, last_activity, high_credit, balance, active, status, overdue, month_payment, status_payment, overdue_count, payments_per_tradeline]
# payments_per_tradeline is x which is tup[16]
# overdue_count is a

def outer(l : list, total: list) -> list:
    elem = l[-1]
    return [elem for x in range(sum(total))]

def tradeline_outer(tradeline_attr: list, total_per_tradeline: list) -> list:
    ret = []
    count = 0
    for tradeline_len in total_per_tradeline:
#         print(tradeline_attr[count])
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

def create_cols(tup : list, flag: str) -> list:
    
    payments_per_tradeline = tup[-1]
    
    
    tup[0] = outer(tup[0], payments_per_tradeline)
    tup[1] = outer(tup[1], payments_per_tradeline)
    tup[2] = outer(tup[2], payments_per_tradeline)
    tup[3] = outer(tup[3], payments_per_tradeline)
    tup[4] = outer(tup[4], payments_per_tradeline)
    
    if flag == 'train':
        tup[5] = outer(tup[5], payments_per_tradeline) # score
    else:
        tup[5] = tradeline_outer(tup[5], payments_per_tradeline)
    

    # 6 - 11
    tup[6] = tradeline_outer(tup[6], payments_per_tradeline)
    tup[7] = tradeline_outer(tup[7], payments_per_tradeline)
    tup[8] = tradeline_outer(tup[8], payments_per_tradeline)
    tup[9] = tradeline_outer(tup[9], payments_per_tradeline)
    tup[10] = tradeline_outer(tup[10], payments_per_tradeline)
    
    if flag == 'train':
        tup[11] = tradeline_outer(tup[11], payments_per_tradeline)

        # 12
        tup[12] = overdue_col(tup[12], tup[15], payments_per_tradeline)
    else:
        tup[11] = overdue_col(tup[11], tup[14], payments_per_tradeline)

    # 13 -- flag
#     payments_per_tradeline = tup[-1]
    tup.pop()
    tup.pop()

    trade_line_f = trade_line_flag(payments_per_tradeline)
    tup.append(trade_line_f)
    
        
    return tup



        
def create_dataframe(file: list, flag : str):
    if flag == 'train':
        cols = ['ID', 'first_name', 'last_name', 'SSN', 'address', 'score', 
                'type', 'last_activity' , 'high_credit', 'balance', 'active', 'status',
                'overdue', 'month_payment', 'status_payment', 'tradeline_flag']
    else:
        cols = ['ID', 'first_name', 'last_name', 'SSN', 'address', 
                'type', 'last_activity' , 'high_credit', 'balance', 'active', 'status',
                'overdue', 'month_payment', 'status_payment', 'tradeline_flag']
    cols_dict = {}
    index = 0
    for l in file:
        cols_dict[cols[index]] = l
        index +=1
    return pd.DataFrame(cols_dict)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
