import pandas as pd
import numpy as np
from factormodel.pre_process import panels2stack

def singlesort_id_t(df_t_stack,g):
    '''
    描述:
    根据t时刻的公司特征(character)和股票状态(state)对股票代码(code)进行分组,输出每个股票对应的组号(1-g),有缺失值或不考虑的股票组号为np.nan

    输入参数:
    df_t_stack:DataFrame,堆栈数据,共3列,列名为:['code','character','state'],shape = [N,3],N为股票数量
        code:str,股票代码
        character:float,用于分组的公司特征
        state:float,股票当日分组时是否考虑(一般不考虑ST股票或新上市股票)
    
    g:int,分组数量

    输出参数:
    sort_id_t:Series,每个股票对应的组号(1-g),缺失值的股票组号为nan,index为股票代码,shape = [N,],N为股票数量
    '''
    # 防止断点时影响外部变量
    df_t = df_t_stack.copy()
    # 提前设置存储sort结果的Series,以code为index并将值全部设为nan,方便后面填入无数据缺失且state正常的股票分组结果
    sort_id_t = pd.Series(np.full(len(df_t),np.nan),index = df_t['code'])
    # 删除有任何数据缺失的股票,缺失过多时不分组,全部股票的分组结果都输出为nan
    df_t.dropna(inplace = True)
    # 去掉分组时不考虑的股票
    df_t = df_t[df_t['state'] == 1].reset_index(drop = True)
    # 当期可交易股票过少时直接返回值为nan的DataFrame
    if len(df_t) <= g*2:
        return sort_id_t
    # 分为g组，共g+1个分割点(包括最大值和最小值)
    percentile = np.percentile(df_t['character'],[100/g*i for i in range(g+1)])
    # 对所有未缺失数据的股票分组，输出这些股票对应的组的序号(从1到g的整数)
    df_t['id'] = pd.cut(df_t['character'],bins = percentile,labels = np.arange(1,g+1))
    # 将上面的有效序号填到sort_id_t中，使缺失数据的归到nan组，未缺失数据的正常分组
    sort_id_t[df_t['code']] = np.array(df_t['id'])
    return pd.DataFrame(sort_id_t)


def singlesort_ret_t(df_sort_t):
    '''
    描述:
    根据t时刻股票收益率、上一期股票分组结果和股票权重计算t时刻分组收益率

    输入参数:
    df_sort_t:DataFrame,堆栈数据,共5列,列名为:['date','code','ret','id','weight']
        date:datetime.date,计算分组收益率当天的日期,所有行的值相等(因为上一步是对date做了groupby)
        code:str,股票代码
        ret:float,当期收益率
        id:float,上一期分组的组别,尽管type(id)为float,但实际全为1-组数的整数
        weight:float,上一期每个股票的收益率权重(例如市值)

    输出参数:
    ret_sort_t:Series,每组收益率,shape = (组数,)
    '''
    def group_ret_cal(df_group):
        return df_group['ret']@df_group['weight']/np.sum(df_group['weight'])
    ret_sort_t = df_sort_t.groupby('id').apply(group_ret_cal)
    ret_sort_t.name = df_sort_t.iloc[0,0]
    return ret_sort_t


def singlesort_stack(df_stack,g,weighted,stated):
    """
    描述：
    提取堆栈数据的信息,并根据信息输出所有时刻单变量分组收益率

    输入参数：
    df_stack:DataFrame,堆栈数据,共6列,列名为['code','date','ret','character',*'weight',*'state']
        date:datetime.date,日期
        code:str,股票代码
        ret:float,当期股票收益率
        character:float,用于分组的公司特征
        *weight:float,计算加权收益率时,每个股票对应的指标(如市值),没有wesight列时计算等权重收益率
        *state:float,股票当日分组时是否考虑(一般不考虑ST股票或新上市股票)
    
    g:int,分组个数
    
    weighted:bool,是否计算加权收益率

    stated:bool,是否对ST或流动性不高的股票进行筛选

    输出参数:
    ret_sort:面板DataFrame,分组收益率,index为datetime.date格式的日期,shape = [T,g]

    df_port:堆栈DataFrame,每个组合中各股票的权重,共4列,列名为['date','code','id','weight']
        date:datetime.date,日期
        code:str,股票代码
        id:float,1~g的整数,分组编号
        weight:float,归一化后的股票权重,相同日期和id的股票weight之和为1
    """
    df = df_stack.copy()# 防止断点时影响外部变量
    # 为了简洁，不计算加权收益率时添加weight列和state并将值全部设为1
    if weighted == True and stated == True:
        df.columns = ['date','code','ret','character','weight','state']
    elif weighted == True and stated == False:
        df.columns = ['date','code','ret','character','weight']
        df['state'] = pd.Series(np.ones(len(df)))
    elif weighted == False and stated == True:
        df.columns = ['date','code','ret','character','state']
        df['weight'] = pd.Series(np.ones(len(df)))
    elif weighted == False and stated == False:
        df.columns = ['date','code','ret','character']
        df['weight'] = pd.Series(np.ones(len(df)))
        df['state'] = pd.Series(np.ones(len(df)))
    # 计算单分组结果并合并入堆栈数据df中
    sort_id = df.groupby('date')[['code','character','state']].apply(lambda x: singlesort_id_t(x,g))
    sort_id = sort_id.reset_index()
    sort_id.columns = ['date','code','id']
    df = pd.concat([df.set_index(['date','code']),sort_id.set_index(['date','code'])],\
               axis = 1,join = 'inner').reset_index()
    df.drop(['state'],axis = 1,inplace = True)
    # 计算上一期分组结果和权重
    df[['id','weight']] = df.groupby('code')[['id','weight']].shift(1)
    df.dropna(inplace = True) #去掉任何存在缺失数据的行
    df = df[['date','code','ret','id','weight']]
    # 计算分组收益率,最终得到的ret_sort的index对应的收益率为下一期的收益率
    try:
        ret_sort = df.groupby('date').apply(singlesort_ret_t).unstack().fillna(0)
    except:
        ret_sort = df.groupby('date').apply(singlesort_ret_t).fillna(0)
    ret_sort.columns = list(range(1,g+1))
    # 计算各期每个组合中各个股票的权重
    df_port = df.drop('ret',axis = 1)
    def weight_normalize(df_weight):
        df_weight['weight'] = df_weight['weight']/sum(df_weight['weight'])
        df_weight = df_weight.set_index('code')
        return df_weight
    df_port = df_port.groupby(['date','id'])[['code','weight']].apply(weight_normalize)
    df_port = df_port.reset_index()
    df_port = df_port[['date','code','id','weight']]
    return ret_sort,df_port


def singlesort_unstack(ret,character,g,weighted,stated,weight = None,state = None):
    """
    描述：
    提取面板数据的信息,并根据信息输出所有时刻单变量分组收益率

    输入参数：
    ret:DataFrame,股票收益率,index为日期(datetime.date),columns为股票代码(str),shape = [T,N]

    character:DataFrame,用于分组的公司特征,index为日期(datetime.date),columns为股票代码(str),shape = [T,N]
    
    g:int,分组个数
    
    weighted:bool,是否计算加权收益率
    
    stated:bool,是否对ST或流动性不高的股票进行筛选

    weight:DataFrame,计算加权收益率时对应的指标(如市值),没有weight时计算等权重收益率,
        index为日期(datetime.date),columns为股票代码(str),shape = [T,N]
    
    state:DataFrame,筛选ST或流动性不高的股票时对应的指标,没有state时不筛选,
        index为日期(datetime.date),columns为股票代码(str),values为0或1(float),空值为nan,shape = [T,N]

    输出参数:
    ret_sort:DataFrame,分组收益率,index为datetime.date格式的日期,shape = [T,g]

    df_port:堆栈DataFrame,每个组合中各股票的权重,共4列,列名为['date','code','id','weight']
        date:datetime.date,日期
        code:str,股票代码
        id:float,1~g的整数,分组编号
        weight:float,归一化后的股票权重,相同日期和id的股票weight之和为1
    """
    if weighted == True and stated == True:
        df_stack = panels2stack([ret,character,weight,state])
    elif weighted == True and stated == False:
        df_stack = panels2stack([ret,character,weight])
    elif weighted == False and stated == True:
        df_stack = panels2stack([ret,character,state])
    elif weighted == False and stated == False:
        df_stack = panels2stack([ret,character])
    ret_sort,df_port = singlesort_stack(df_stack,g,weighted,stated)
    return ret_sort,df_port


def long_short_cal(ret,df_port,long_only = False,fee = None):
    """
    描述:根据分组收益率计算多空组合收益率

    输入参数：
    ret:DataFrame,分组收益率,index为datetime.date格式的日期,shape = [T,group]

    df_port:堆栈DataFrame,每个组合中各股票的权重,共4列,列名为['date','code','id','weight']
        date:datetime.date,日期
        code:str,股票代码
        id:float,1~g的整数,分组编号
        weight:float,归一化后的股票权重,相同日期和id的股票weight之和为1

    long_only:bool,long_only == True时只计算多头收益,不计算多空收益,long_only == False时计算多空收益
        
    fee:float,交易费率(一般用0.003),fee == None时不考虑交易费用

    输出参数：
    df_long_short:DataFrame,index为datetime.date格式的日期,第一列为多空组合收益率,shape = [T,1]
    """
    if long_only == False and fee == None:
        long_short_ret = ret.iloc[:,-1] - ret.iloc[:,0]
    elif long_only == True and fee == None:
        long_short_ret = ret.iloc[:,-1]
    elif long_only == False and fee != None:
        # 只保留long的部分和short的部分
        long_id = max(df_port['id'])
        short_id = min(df_port['id'])
        df_port = df_port[(df_port['id'] == short_id)|(df_port['id'] == long_id)]
        # 计算多空组合中各个股票的权重
        df_port = pd.pivot(data = df_port,index = ['date','id'],\
                                columns = 'code',values = 'weight').fillna(0)
        # 计算多空组合股票权重(多头权重序列-空头权重序列)
        df_port_ = df_port.groupby('date').apply(\
        lambda port_t: (port_t - port_t.shift(1)).dropna().reset_index(drop = True))
        df_port_ = df_port_.droplevel(1)# 删掉多余的index
        # 计算当期多空权重相比上一期多空权重变化的绝对值
        df_port_change = np.sum(np.abs((df_port_ - df_port_.shift(1)).dropna()),axis = 1)
        long_short_ret = ret.iloc[:,-1] - ret.iloc[:,0]
        ret_fee = pd.concat([long_short_ret,df_port_change*fee],axis = 1).fillna(0)
        long_short_ret = ret_fee[0] - ret_fee[1]
    else:
        # 只保留long的部分
        long_id = max(df_port['id'])
        df_port = df_port[df_port['id'] == long_id]
        # 计算多头组合中各个股票的权重
        df_port = pd.pivot(data = df_port,index = 'date',\
                                columns = 'code',values = 'weight').fillna(0)
        # 计算当期多头权重相比上一期多头权重变化的绝对值
        df_port_change = np.sum(np.abs((df_port - df_port.shift(1)).dropna()),axis = 1)
        long_short_ret = ret.iloc[:,-1]
        ret_fee = pd.concat([long_short_ret,df_port_change*fee],axis = 1).fillna(0)
        long_short_ret = ret_fee[0] - ret_fee[1]
    # 为了适应下面算累计多空净值，这里刻意不输出Series而是Dataframe，以保证结果有两个维度
    df_long_short = pd.DataFrame(long_short_ret,index = ret.index,columns=['long_short_ret'])
    df_long_short.index.name = 'date'
    return df_long_short


def net_val_cal(ret,figname,show = False):
    """
    描述:根据分组收益率计算各组累计净值或多空组合净值并画图

    输入参数:
    ret:面板DataFrame,分组收益率,index为datetime.date格式的日期,shape = [T,group]
    
    figname:str,图片名称

    show:bool,是否显示图像,当show == True时,会在本函数处暂停向下运行并显示分组净值图像；
    
    show == False时不显示函数图像,只在主函数路径下导出cumulative net value.jpg的文件

    输出参数：
    cum_ret:DataFrame,分组累计净值,index为日期,第一列到最后一列为各个日期的分组净值,shape = [T,group]
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
    plt.rcParams['axes.unicode_minus'] = False
    group = np.size(ret,1)
    cum_ret = (1 + ret).copy()
    for t in range(len(cum_ret) - 1):
        cum_ret.iloc[t + 1,:] = cum_ret.iloc[t,:]*cum_ret.iloc[t + 1,:] #计算累计净值
    cum_ret = cum_ret/cum_ret.iloc[0,:]
    plt.figure(figname,figsize=(10,10))
    if group == 1: #只有1列，默认该列为多空组合收益率，计算多空组合累计净值
        plt.plot(cum_ret.iloc[:,0],label = '多空组合净值')
        plt.legend()
    else:
        for i in range(group):
            plt.plot(cum_ret.iloc[:,i],label = '第{}组净值'.format(i + 1))
            plt.legend()
    plt.savefig('{}.jpg'.format(figname))
    if show == True:
        plt.show()
    else:
        pass
    plt.close()
    return cum_ret