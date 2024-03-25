import pandas as pd
import numpy as np
from factormodel.pre_process import standardize_stack


def ic_cal_stack(df_stack,ic_lst,type = 'rank'):
    '''
    描述:
    根据堆栈收益率和因子数据批量算因子normal ic和rank ic

    输入变量:
    df_stack:堆栈DataFrame,第一列为datetime.date类型的日期,第二列为str类型的股票代码,
        其它列是收益率及公司特征值,shape = [T*N,2+n]
    
    ic_lst:list,计算涉及的变量名,ic_lst[0]为当期收益率列名,其余为相关因子列名

    type:str,计算的IC类型,'normal'或'rank'

    输出变量:
    df_ic:面板DataFrame,index为日期,columns为因子名,values为各期因子ic
    '''
    df_ = df_stack.copy()
    datename = df_.columns[0]
    codename = df_.columns[1]
    retname = ic_lst[0]
    ## 计算下一期收益率
    df_['nxtret'] = df_.groupby(codename)[retname].shift(-1)
    df_.dropna(how = 'any',subset = ['nxtret'],inplace = True)
    df_ic = df_.pivot(index = datename,columns = codename,values = 'nxtret')
    ## 计算各期IC
    method = 'spearman' if type == 'rank' else 'pearson'
    for factorname in ic_lst[1:]:
        df_ic[factorname] = df_.groupby(datename).apply(\
            lambda x: x['nxtret'].corr(x[factorname],method = method))
    df_ic = df_ic[ic_lst[1:]]
    return df_ic
    

def icir_cal(df_ic):
    '''
    描述:
    根据因子IC序列计算IC均值和IR

    输入变量:
    df_ic:面板DataFrame,index为日期,columns为因子名,values为各期因子ic值,shape = [T,n]

    输出变量:
    icmean:Series,各因子ic均值,shape = [n],n == 1时type(icmean) = float

    ir:Series,各因子ir,shape = [n],n == 1时type(ir) = float
    '''
    icmean = np.nanmean(df_ic)
    ir = icmean/np.nanstd(df_ic)
    return icmean,ir


def ratios_cal(df_ratio_pre,multi):
    '''
    描述:
    根据收益率和累计收益率序列计算年化夏普比率,最大回撤率

    输入变量:
    df_ratio_pre:面板DataFrame,index为日期,第一列为收益率,第二列为累计收益率,第三列为无风险利率

    multi:计算年化夏普比率时的乘数,如日化转年化sqrt(252),月化转年化sqrt(12),周化转年化sqrt(52)

    输出变量:
    sharp,max_drawdown:float,年化夏普和最大回撤率
    '''
    df_ratio_pre.columns = ['ret','cum_ret','rf']
    df_ratio = df_ratio_pre.dropna(how = 'any')
    sharp = np.nanmean(df_ratio['ret'] - df_ratio['rf'])/np.nanstd(df_ratio['ret'])*multi
    # 各期的回撤率
    drawdown = (np.maximum.accumulate(df_ratio['cum_ret']) - df_ratio['cum_ret'])/np.maximum.accumulate(df_ratio['cum_ret'])
    # 最大回撤率
    max_drawdown = max(drawdown)
    return sharp,max_drawdown


def newey_west_test(df_ret,lag = None):
    """
    描述：
    计算收益率序列的均值,newey-west t值

    输入变量:
    df_ret:DataFrame,收益率时间序列,index为datetime.date格式的日期,各列为收益率,shape = [T,n]

    lag:int,Newey-West滞后阶数,默认为int(4*(T/100)^(2/9))

    输出变量：
    mean_ret:df_ret列数 > 1时为list,各列时序收益率均值;df_ret列数 = 1时为float,时序收益率均值

    tval:df_ret列数 > 1时为list,各列时序收益率的Newey-west t值;df_ret列数 = 1时为float,时序收益率的Newey-west t值
    """
    import statsmodels.formula.api as smf
    if lag == None:
        lag = int(4*(len(df_ret)/100)**(2/9))
    # 重命名列名,保证其符合变量名规范
    colnum = np.size(df_ret,1)
    colrename = []
    for i in range(colnum):
        colrename.append('col' + str(df_ret.columns[i]))
    df_ret.columns = colrename
    # 根据列数判断计算方式,列数大于1时循环各列计算
    if colnum > 1:
        mean_ret = []
        tval = []
        for i in range(colnum):
            colname = df_ret.columns[i]
            reg = smf.ols('{}~1'.format(colname),data = df_ret).fit(cov_type = 'HAC',cov_kwds = {'maxlags':lag})
            mean_ret.append(reg.params[0])
            tval.append(reg.tvalues[0])
        return mean_ret,tval
    else:
        colname = df_ret.columns[0]
        reg = smf.ols('{}~1'.format(colname),data = df_ret).fit(cov_type = 'HAC',cov_kwds = {'maxlags':lag})
        mean_ret = reg.params[0]
        tval = reg.tvalues[0]
        return mean_ret,tval


def newey_west_reg(ret_factor,lag = None):
    """
    描述：
    将股票收益率或组合收益率向因子收益率回归,计算回归的alpha,beta,newey-west t值

    输入变量:
    ret_factor:DataFrame,第一列为股票或组合收益率,第二列到最后一列为因子收益率,shape = [T,1 + factor_num]

    lag:int,Newey-West滞后阶数,默认为int(4*(T/100)^(2/9))

    输出变量：
    param:list,回归系数,当alpha_cal == True时计算包含截距,否则不包含截距
    
    tval:list,回归系数的Newey-west t值
    """
    import statsmodels.formula.api as smf
    ret_factor.dropna(inplace = True)
    if lag == None:
        lag = int(4*(len(ret_factor)/100)**(2/9))
    N = np.size(ret_factor,axis = 1) - 1 #因子数量
    column_lst = ['ret']
    reg_equation = 'ret~'
    for i in range(N):
        columni = 'factor' + str(i + 1)
        column_lst.append(columni)
        reg_equation = reg_equation + columni + '+'
    reg_equation = reg_equation[:-1]
    ret_factor.columns = column_lst
    if len(ret_factor) == 0: #去除空值后若DataFrame为空，则直接返回空值
        idx = ['Intercept']
        idx = idx + column_lst[1:]
        nanreturn = pd.Series(np.full([N + 1,],np.nan),index = idx)
        return nanreturn,nanreturn,nanreturn #这里保持格式和index一致是为了防止在apply中使用该函数导致拼接结果出错
    reg = smf.ols(reg_equation,data = ret_factor).fit(cov_type = 'HAC',cov_kwds = {'maxlags':lag})
    params = list(reg.params)
    tvals = list(reg.tvalues)
    return params,tvals


def beta_rolling(df,p,rolling_w):
    '''
    描述：
    根据股票收益率和风险因子时间序列数据,滚动进行newey-west回归,并计算股票在各期风险因子上的风险暴露(beta)

    输入变量:
    df:面板DataFrame,index为日期(datetime.date),columns为股票代码(str),\
        index为日期(datetime.date),columns为股票代码(str),\
        前N列为不同代码的股票各期收益率,后p列为不同风险因子的时序数据,shape = [T,N + p]
    
    p:风险因子数量
    
    rolling_w:回归窗口

    输出变量:
    df_beta_lst:list,shape = [p],每个元素是一个DataFrame
        第i个DataFrame的t行j列是t时刻风险因子i在股票j上的滚动beta值,shape = [T - rolling_w,p]
    '''
    N = np.size(df,1) - p #N个股票
    T = len(df) #T个时间点
    # 每个风险因子一张DataFrame,共p张DataFrame,对应该风险因子的滚动beta值
    df_beta_0 = pd.DataFrame(np.zeros([T - rolling_w,N]),\
                 index = df.index[rolling_w:],columns = df.columns[:-p])
    # df_beta_0是模板,根据df_beta_0生成p个shape一致的DataFrame
    for i in range(1,p + 1):
        varname = 'df_beta_' + str(i)
        vars()[varname] = df_beta_0.copy()
    for t in range(rolling_w,T): #滚动计算过去rolling_w的beta
        df_stock_t = df.iloc[t - rolling_w:t,:-p]
        df_factor_t = df.iloc[t - rolling_w:t,-p:]
        # 对df_stockt的每列apply,分别计算每个股票的newey-west beta
        def beta_cal(df_stock_ti,df_factor_t):
            ret_factor_ti = pd.merge(df_stock_ti,df_factor_t,how = 'left',on = 'date')
            params_i,tvals_i,pvals_i = newey_west_reg(ret_factor_ti)
            return params_i
        df_beta_t = df_stock_t.apply(lambda x: beta_cal(x,df_factor_t))
        # df_beta_t共p+1行,每一列是一只股票在p个因子上的截距和beta值
        for i in range(1,p + 1):
            varname = 'df_beta_' + str(i)
            vars()[varname].iloc[t - rolling_w,:] = df_beta_t.iloc[i,:]
    df_beta_lst = [vars()['df_beta_' + str(i)] for i in range(1,p + 1)]
    return df_beta_lst


def fama_macbeth(df_stack,formula_lst,lag = None):
    '''
    描述:
    对堆栈数据(pretty_stack,panels2stack,del_outlier_stack输出格式均可)
        做Fama-MacBeth回归并输出fama_macbeth对象

    输入变量:
    df_stack:堆栈DataFrame,第一列为datetime.date类型的日期,第二列为str类型的股票代码,
        其它列是收益率及公司特征值,shape = [T*N,2+n]
    
    formula_lst:list,回归方程中的变量名,formula_lst[0]为当期收益率列名,其余为相关因子列名

    lag:int,Newey-West滞后阶数,默认为int(4*(T/100)^(2/9))

    输出变量:
    fm:fama_macbeth对象(linearmodels.FamaMacBeth)
    '''
    from linearmodels import FamaMacBeth
    df_fm = df_stack.copy()
    datename = df_fm.columns[0]
    codename = df_fm.columns[1]
    retname = formula_lst[0]
    T = len(set(df_fm[datename]))
    ## 如果没有指定,需要计算默认滞后阶数
    if lag == None:
        lag = int(4*(T/100)**(2/9))
    ## 特征标准化
    df_fm = standardize_stack(df_fm,formula_lst[1:],type = 'standardize')
    ## 计算下一期收益率
    df_fm['nxtret'] = df_fm.groupby(codename)[retname].shift(-1)
    ## 调整date类型
    df_fm[datename] = pd.to_datetime(df_fm[datename])
    ## 设置双重index,注意索引顺序,删掉影响FM回归的空值
    df_fm.set_index([codename,datename],inplace = True)
    df_fm.dropna(subset = formula_lst[1:].append('nxtret'),how = 'any',inplace = True)
    ## 写出回归方程
    formula = 'nxtret~1+'
    for varname in formula_lst[1:]:
        formula = formula + varname + '+'
    formula = formula[:-1]
    ## Fama-MacBeth回归
    mod = FamaMacBeth.from_formula(formula,data = df_fm)
    fm = mod.fit(cov_type = 'kernel',debiased = False,bandwidth = lag)
    return fm