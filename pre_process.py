import pandas as pd
import numpy as np
from scipy.stats import mode

def pretty_unstack(df_unstack_lst,format_lst):
    '''
    描述:
    对一系列面板数据格式标准化,将日期格式标准化datetime.date并置为index,将各dataframe的日期对应一致

    输入变量:
    df_unstack_lst:list,需要处理的面板数据列表,其中的每个元素是面板DataFrame,shape = [X,1]
        DataFrame的第一列为str类型的日期,其它列是因子或公司特征值,shape = [T,1+N]
    
    format_lst:list,各个面板数据对应的str类型的日期格式,如'%Y-%m-%d'等,其中每个元素为str

    输出变量:
    df_pretty_lst:list,其中的每个元素是标准化后的面板DataFrame,shape = [X,1]
        DataFrame的index为datetime.date类型的日期,index_name = 'date'
        各列是因子或公司特征值,shape = [T,N]
    '''
    dftemp_lst = []
    df_pretty_lst = []
    start_lst = []
    end_lst = []
    import datetime as dt
    num = len(format_lst) #面板数据个数
    for i in range(num):
        # 调整日期格式
        df_pre = df_unstack_lst[i]
        df_ = df_pre.copy()
        date_format = format_lst[i]
        if type(df_.iloc[0,0]) == str: # 如果第一列已经是datetime类型的数据了,无需调整
            df_.iloc[:,0] = df_pre.iloc[:,0].apply(lambda x : dt.datetime.strptime(x,date_format).date())
        # 重新设置索引
        index_name = df_.columns[0]
        df_.set_index(keys = index_name,inplace = True)
        df_.index.name = 'date'
        df_ = df_.dropna(how = 'all')
        dftemp_lst.append(df_)
        # 提取各dataframe的起始日期和结束日期
        start_lst.append(df_.index[0])
        end_lst.append(df_.index[-1])
    # 取各dataframe的日期交集
    start = max(start_lst)
    end = min(end_lst)
    for i in range(num):
        df_temp = dftemp_lst[i]
        df_temp  = df_temp[(df_temp.index >= start)*(df_temp.index <= end)]
        df_pretty_lst.append(df_temp)
    return df_pretty_lst


def pretty_stack(df_stack,date_format):
    '''
    描述:
    堆栈数据格式标准化,将日期格式标准化datetime.date,去除特征全为空值的日期

    输入变量:
    df_stack:DataFrame,第一列为str类型的日期,第二列为str类型的股票代码,其它列是公司特征值,shape = [T*N,2+n]
    
    date_format:str,日期格式,如'%Y-%m-%d'等

    输出变量:
    df_pretty:DataFrame,第一列为datetime.date类型的日期,第二列str类型的股票代码,
        columns = ['date','code',...],其余各列是公司特征值,shape = [T*N,2+n]
    '''
    import datetime as dt
    datename = df_stack.columns[0]
    codename = df_stack.columns[1]
    # 提取特征全为空值的日期,删除该日期数据
    def find_nulldate(df_t):# df_t是某日期不同股票的特征堆栈数据,判断某日期是否全为空值
        if sum(np.sum(~pd.isna(df_t.iloc[:,2:]))) == 0: #该日期的特征全为空
            return 1
        else:
            return 0
    df_nulldate = df_stack.groupby(datename, group_keys = False).apply(find_nulldate)
    nulldates = df_nulldate[df_nulldate == 1].index
    # 删除指定日期
    df_pretty_pre = df_stack[~df_stack[datename].isin(nulldates)]
    df_pretty_pre.reset_index(drop = True,inplace = True)
    df_pretty = df_pretty_pre.copy()
    # 调整日期格式
    if type(df_pretty.iloc[0,0]) == str: # 如果第一列已经是datetime类型的数据了,无需调整
        df_pretty.iloc[:,0] = df_pretty_pre.iloc[:,0].apply(lambda x : dt.datetime.strptime(x,date_format).date())
    # 重新设置前两列列名(date,code)
    df_pretty.rename(columns = {datename:'date',codename:'code'},inplace = True)
    return df_pretty


def panels2stack(df_unstack_lst,colname_lst = None):
    '''
    描述:
    将一系列格式标准化后的面板数据(经过pretty_unstack处理后)合并并转化为堆栈数据

    输入变量:
    df_unstack_lst:list,其中的每个元素是标准化后的面板DataFrame,shape = [X,1]
        DataFrame的index为datetime.date类型的日期,index_name = 'date'
        各列是因子或公司特征值,shape = [T,N]

    colname_lst:输出的堆栈数据列名(除了'date'和'code'),colname_lst = None时不设置列名
    
    输出变量:
    df:重新设置index后的堆栈数据,其第一列为date,第二列为code
    '''
    df_stack = pd.concat([df_i.stack() for df_i in df_unstack_lst],axis = 1)
    df_stack.reset_index(inplace = True)
    if colname_lst != None:
        df_stack.columns = ['date','code'] + colname_lst
    return df_stack


def stack2panels(df_stack,idxname,colname,panelname_lst):
    '''
    描述:
    将格式标准化后的堆栈数据(经过pretty_stack处理后)批量转化为一系列面板数据

    输入变量:
    df_stack:堆栈DataFrame,index为普通index(1,2,3...),列名中包含idxname,colname,panelname_lst

    idxname:生成的面板DataFrame的index名

    colname:生成的面板DataFrame的column名

    panelname_lst:原堆栈数据中需要做反堆栈的列名

    输出变量:
    df_unstack_lst:list,其中每个元素是面板DataFrame,shape = [len(panelname_lst),1],
        DataFrame的index名为idxname,column名为colname,shape = [len(colname),len(idxname)],
        值为原数据以panelname_lst中值为列名的值
    '''
    df_unstack_lst = []
    for i in range(len(panelname_lst)):
        panel_temp = df_stack.pivot(index = idxname,\
                    columns = colname,values = panelname_lst[i])
        df_unstack_lst.append(panel_temp)
    return df_unstack_lst


def del_outlier_unstack(df_unstack,method,arg):
    '''
    描述：
    对面板数据进行去极值,将超出范围的数据改为临界点

    输入变量：
    df_unstack:面板DataFrame,index为datetime.date类型的日期,各列是因子或公司特征值,shape = [T,N]
    
    method:str,去极值方式,共3种:{'mad','sigma','percentile'}
        mad:根据数据中位数去极值
        sigma:根据数据方差去极值
        percentile:根据数据分位点去极值
    
    arg:list,根据method确定的参数
        method == mad:arg = [n1(mad方法左侧极值参数),n2(mad方法右侧极值参数)]
        method == sigma:arg = [n1(sigma方法左侧极值参数),n2(sigma方法右侧极值参数)]
        method == percentile:arg = [percentile1(float,不带百分号的percentile方法百分位数,左侧极值参数)\
        percentile2(float,不带百分号的percentile方法百分位数,右侧极值参数)]

    输出变量:
    df_del:面板DataFrame,去极值后的数据
    '''
    if method.lower() == 'mad':
        df_del = df_unstack.apply(mad_del,name = None,lst_n = arg,axis = 1)
    elif method.lower() == 'sigma':
        df_del = df_unstack.apply(sigma_del,name = None,lst_n = arg,axis = 1)
    elif method.lower() == 'percentile':
        df_del = df_unstack.apply(percentile_del,name = None,lst_percentile = arg,axis = 1)
    return df_del


def del_outlier_stack(df_stack,name,method,arg):
    '''
    描述：
    对堆栈的DataFrame进行去极值,将超出范围的数据改为临界点

    输入变量：
    df_stack:堆栈的DataFrame,第一列为日期(datetime.date),第二列为股票代码(str),shape = [T*N,2 + n]
    
    name:str,需要去极值的列名
    
    method:str,去极值方式,共3种:{'mad','sigma','percentile'}
        mad:根据数据中位数去极值
        sigma:根据数据方差去极值
        percentile:根据数据分位点去极值
    
    arg:list,根据method确定的参数
        method == mad:arg = [n1(mad方法左侧极值参数),n2(mad方法右侧极值参数)]
        method == sigma:arg = [n1(sigma方法左侧极值参数),n2(sigma方法右侧极值参数)]
        method == percentile:arg = [percentile1(float,不带百分号的percentile方法百分位数,左侧极值参数)\
        percentile2(float,不带百分号的percentile方法百分位数,右侧极值参数)]

    输出变量:
    df_del:DataFrame,去极值后的数据
    '''
    datename = df_stack.columns[0]
    if method.lower() == 'mad':
        df_del = df_stack.groupby(datename, group_keys = False).apply(mad_del,name = name,lst_n = arg)
    elif method.lower() == 'sigma':
        df_del = df_stack.groupby(datename, group_keys = False).apply(sigma_del,name = name,lst_n = arg)
    elif method.lower() == 'percentile':
        df_del = df_stack.groupby(datename, group_keys = False).apply(percentile_del,name = name,lst_percentile = arg)
    return df_del


def mad_del(df, name, lst_n):
    '''
    描述:
    对某个日期的截面数据采用单期MAD法去极值

    输入变量:
    df:DataFrame
        name != None时第一列为N个股票代码(str),每行是n+个需要去极值的公司特征,shape = [N,1 + n]
        name == None时每行为N个股票代码(str),shape = [1,N]

    name:str,需要去极值的列名,name == None时是对单期面板去极值
    
    lst_n:list:[n1(mad方法左侧极值参数),n2(mad方法右侧极值参数)]
        mad方法极值参数为中位数设置位置,阈值设置在几个绝对偏差中位数上
    
    输出变量:
    df:去除极值后的DataFrame
    '''
    n1 = lst_n[0]
    n2 = lst_n[1]
    if name != None:
        factor_median = np.nanmedian(df[name],axis = 0)
        bias_sr = abs(df[name] - factor_median)
        new_median = np.nanmedian(bias_sr,axis = 0)
        dt_down = factor_median - n1 * new_median
        dt_up = factor_median + n2 * new_median
        df[name] = df[name].clip(dt_down,dt_up,axis = 1)
    else:
        factor_median = np.nanmedian(df)
        bias_sr = abs(df - factor_median)
        new_median = np.nanmedian(bias_sr)
        dt_down = factor_median - n1 * new_median
        dt_up = factor_median + n2 * new_median
        df = df.clip(dt_down,dt_up)
    return df


def sigma_del(df, name, lst_n):
    '''
    描述:
    对某个日期的截面数据采用单期sigma法去极值

    输入变量:
    df:DataFrame
        name != None时第一列为N个股票代码(str),每行是n+个需要去极值的公司特征,shape = [N,1 + n]
        name == None时每行为N个股票代码(str),shape = [1,N]
    
    name:str,需要去极值的列名,name == None时是对单期面板去极值
    
    lst_n:list:[n1(sigma方法左侧极值参数),n2(sigma方法右侧极值参数)]
        sigma方法极值参数为标准差设置位置,阈值设置在几个标准差的地方,一般为3个标准差(n1=n2=3)

    输出变量:
    df:去除极值后的DataFrame
    '''
    n1 = lst_n[0]
    n2 = lst_n[1]
    if name != None:
        mean = np.nanmean(df[name],axis = 0)
        std = np.nanstd(df[name],axis = 0)
        dt_down = mean - n1 * std
        dt_up = mean + n2 * std
        df[name] = df[name].clip(dt_down,dt_up,axis = 1)
    else:
        mean = np.nanmean(df)
        std = np.nanstd(df)
        dt_down = mean - n1 * std
        dt_up = mean + n2 * std
        df = df.clip(dt_down,dt_up)
    return df


def percentile_del(df, name, lst_percentile):
    '''
    描述:
    对某个日期的截面数据采用单期percentile法去极值

    输入变量:
    df:DataFrame
        name != None时第一列为N个股票代码(str),每行是n+个需要去极值的公司特征,shape = [N,1 + n]
        name == None时每行为N个股票代码(str),shape = [1,N]
    
    name:str,需要去极值的列名,name == None时是对单期面板去极值
    
    lst_percentile:list:[percentile1(float,percentile方法左侧极值参数)\
        percentile2(float,percentile方法右侧极值参数)]
        percentile方法百分位数为0到100的数

    输出变量:
    df:去除极值后的DataFrame
    '''
    percentile1 = lst_percentile[0]
    percentile2 = lst_percentile[1]
    if name != None:
        dt_up = np.nanpercentile(df[name],percentile2,axis = 0)
        dt_down = np.nanpercentile(df[name],percentile1,axis = 0)
        df[name] = df[name].clip(dt_down,dt_up,axis = 1)
    else:
        dt_up = np.nanpercentile(df,percentile2)
        dt_down = np.nanpercentile(df,percentile1)
        df = df.clip(dt_down,dt_up)
    return df


def standardize_t(df_t,type):
    '''
    描述:
    对相同日期不同股票或因子值的面板数据标准化或归一化
    
    输入变量:
    df_t:Series,某日期股票或因子值,shape = [N,]

    type:str,type == 'standardize'时做标准化(z-score),type == 'normalize'时做归一化

    输出变量:
    df_standard_t:Series,某日期标准化后的股票或因子值,shape = [N,]
    '''
    if sum(~pd.isna(df_t)) != 0: #当日数据不全是空值
        if type == 'standardize':
            df_standard_t = (df_t - np.nanmean(df_t))/np.nanstd(df_t)
        else:
            df_standard_t = (df_t - np.nanmin(df_t))/(np.nanmax(df_t) - np.nanmin(df_t))
    else:
        df_standard_t = df_t
    return df_standard_t


def standardize_unstack(df_unstack,type = 'standardize'):
    '''
    描述:
    股票或因子值的面板数据标准化或归一化

    输入变量:
    df_unstack:面板DataFrame,index为时间,columns为股票代码或因子,shape = [T,N]

    type:str,type == 'standardize'时做标准化(z-score),type == 'normalize'时做归一化

    输出变量:
    df_standard:面板DataFrame,标准化或归一化后的股票或因子值,index为时间,columns为股票代码或因子,shape = [T,N]
    '''
    df_standard = df_unstack.apply(standardize_t,type = type)
    return df_standard


def standardize_stack(df_stack,colname_lst,type = 'standardize'):
    '''
    描述:
    股票或因子值的堆栈数据标准化或归一化

    输入变量:
    df_stack:堆栈DataFrame,第一列是日期,第二列是股票代码或因子,其余n列是需要标准化的股票或因子值,shape = [T*N,2+n]

    colname_lst:需要标准化或归一化的列名

    type:str,type == 'standardize'时做标准化(z-score),type == 'normalize'时做归一化

    输出变量:
    df_standard:堆栈DataFrame,标准化或归一化后的股票或因子值,index为时间,columns为股票代码或因子,shape = [T,N]
    '''
    df_standard = df_stack.copy()
    datename = df_standard.columns[0]
    for colname in colname_lst:
        df_standard[colname] = df_standard.groupby(datename,group_keys = False)[colname].apply(standardize_t,type = type)
    return df_standard


def resamplefreq_unstack(df_unstack,cycle,type_lst):
    '''
    描述:
    将面板较高频率的数据低频化(日度->周度,月度,季度)

    输入变量:
    df_unstack:反堆栈Dataframe,index为日期,columns为需要改变频率的数据,shape = [T,n]

    cycle:int,低频化时的取样周期,例如cycle = 5时重新取样的数据根据原始的5条数据(window = 5)计算出
        常见的cycle = 5(交易日日度->交易日周度),cycle = 20(交易日日度->交易日月度)

    type_lst:list,低频化时各列数据的处理方法,values有{'end'(取窗口最后一个值),'start'(取窗口第一个值),'median'(取窗口数据中位数)
        'mean'(取窗口数据平均值),'common'(取窗口数据众数),'sum'(取窗口数据之和),'max'(取窗口数据最大值),'min'(取窗口数据最小值)}

    输出变量:
    df_resample:反堆栈DataFrame,index为日期,columns为改变数据频率之后的数据,shape = [T,n/cycle]
    '''
    df = df_unstack.copy()
    col_lst = []
    for j in range(np.size(df,1)):
        if type_lst[j] == 'start':# values用cycle开始的值,index用cycle结束的值
            df_temp = df.iloc[range(0,len(df_unstack)-cycle+1,cycle),j]
            df_temp.index = df.index[range(cycle-1,len(df_unstack),cycle)]
        elif type_lst[j] == 'end':# values和index都用cycle结束的值
            df_temp = df.iloc[range(cycle-1,len(df_unstack),cycle),j]
        elif type_lst[j] == 'median':
            df_temp = df.iloc[:,j].rolling(cycle).median().iloc[range(cycle-1,len(df_unstack),cycle)]
        elif type_lst[j] == 'mean':
            df_temp = df.iloc[:,j].rolling(cycle).mean().iloc[range(cycle-1,len(df_unstack),cycle)]
        elif type_lst[j] == 'sum':
            df_temp = df.iloc[:,j].rolling(cycle).sum().iloc[range(cycle-1,len(df_unstack),cycle)]
        elif type_lst[j] == 'max':
            df_temp = df.iloc[:,j].rolling(cycle).max().iloc[range(cycle-1,len(df_unstack),cycle)]
        elif type_lst[j] == 'min':
            df_temp = df.iloc[:,j].rolling(cycle).min().iloc[range(cycle-1,len(df_unstack),cycle)]
        elif type_lst[j] == 'common': #如果存在多个众数,会返回最小的众数
            df_temp = df.iloc[:,j].rolling(cycle).apply(lambda x: mode(x,keepdims = True)[0][0]).\
                iloc[range(cycle-1,len(df_unstack),cycle)]
        col_lst.append(pd.DataFrame(df_temp))
    df_resample = pd.concat(col_lst,axis = 1)
    return df_resample


def resamplefreq_stack(df_stack,cycle,type_lst):
    '''
    描述:
    将堆栈较高频率的数据低频化(日度->周度,月度,季度)

    输入变量:
    df_unstack:堆栈Dataframe,第一列为日期,第二列为股票代码,
        columns = ['date','code',...],shape = [T*N,2 + n]

    cycle:int,低频化时的取样周期,例如cycle = 5时重新取样的数据根据原始的5条数据(window = 5)计算出
        常见的cycle = 5(交易日日度->交易日周度),cycle = 20(交易日日度->交易日月度)

    type_lst:list,低频化时各列数据的处理方法,values有{'end'(取窗口最后一个值),'start'(取窗口第一个值),'median'(取窗口数据中位数)
        'mean'(取窗口数据平均值),'common'(取窗口数据众数),'sum'(取窗口数据之和),'max'(取窗口数据最大值),'min'(取窗口数据最小值)}

    输出变量:
    df_resample:堆栈DataFrame,第一列为日期,第二列为股票代码,shape = [T*N,2 + n/cycle]
    '''
    df = df_stack.copy()
    cal_cols = list(df.columns[2:])
    def group_resample(df_unstack,cycle,type_lst):
        # 调整groups的格式,方便直接用反堆栈的降频方法
        df_unstack = df_unstack.set_index('date')
        df_resample_temp = resamplefreq_unstack(df_unstack,cycle,type_lst)
        return df_resample_temp.reset_index()
    df_resample = df.groupby('code')[['date'] + cal_cols].apply(lambda x: group_resample(x,cycle,type_lst))
    df_resample = df_resample.reset_index()
    # 还原堆栈数据列名顺序
    df_resample = df_resample[['date','code'] + cal_cols]
    return df_resample