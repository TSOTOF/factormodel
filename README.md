# 这是一个面向过程的因子回测框架，包含数据预处理，标准化，单分组，分析交易状态，构建多空组合，风险调整和FM回归(Barra)等功能
# 整体分为3部分：pro process，factor singlesort和factor test

## 1.pre process
### 1.1.pretty_unstack(df_unstack_lst,format_lst)
```python
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
```
### 1.2.pretty_stack(df_stack,date_format)
```python
描述:
堆栈数据格式标准化,将日期格式标准化datetime.date,去除特征全为空值的日期

输入变量:
df_stack:DataFrame,第一列为str类型的日期,第二列为str类型的股票代码,其它列是公司特征值,shape = [T*N,2+n]

date_format:str,日期格式,如'%Y-%m-%d'等

输出变量:
df_pretty:DataFrame,第一列为datetime.date类型的日期,第二列str类型的股票代码,
    columns = ['date','code',...],其余各列是公司特征值,shape = [T*N,2+n]
```
### 1.3.panels2stack(df_unstack_lst,colname_lst = None)
```python
描述:
将一系列格式标准化后的面板数据(经过pretty_unstack处理后)合并并转化为堆栈数据

输入变量:
df_unstack_lst:list,其中的每个元素是标准化后的面板DataFrame,shape = [X,1]
    DataFrame的index为datetime.date类型的日期,index_name = 'date'
    各列是因子或公司特征值,shape = [T,N]

colname_lst:输出的堆栈数据列名,colname_lst = None时不设置列名

输出变量:
df:重新设置index后的堆栈数据,其第一列为date,第二列为code
```
### 1.4.stack2panels(df_stack,idxname,colname,panelname_lst)
```python
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
```
### 1.5.del_outlier_unstack(df_unstack,method,arg)
```python
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
    method == percentile:arg = [percentile1(float,不带百分号的percentile方法百分位数,左侧极值参数)
    percentile2(float,不带百分号的percentile方法百分位数,右侧极值参数)]

输出变量:
df_del:面板DataFrame,去极值后的数据
```
### 1.6.del_outlier_stack(df_stack,name,method,arg)
```python
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
    method == mad:arg = [n1(mad方法左侧极值参数),n2(mad方法右侧极值参数)]\
    method == sigma:arg = [n1(sigma方法左侧极值参数),n2(sigma方法右侧极值参数)]\
    method == percentile:arg = [percentile1(float,不带百分号的percentile方法百分位数,左侧极值参数)\
    percentile2(float,不带百分号的percentile方法百分位数,右侧极值参数)]

输出变量:
df_del:DataFrame,去极值后的数据
```
### 1.7.mad_del(df, name, lst_n)
```python
描述:
对某个日期的截面数据采用单期MAD法去极值

输入变量:
df:DataFrame
    name != None时第一列为N个股票代码(str),每行是n+个需要去极值的公司特征,shape = [N,1 + n]\
    name == None时每行为N个股票代码(str),shape = [1,N]

name:str,需要去极值的列名,name == None时是对单期面板去极值

lst_n:list:[n1(mad方法左侧极值参数),n2(mad方法右侧极值参数)]\
    mad方法极值参数为中位数设置位置,阈值设置在几个绝对偏差中位数上

输出变量:
df:去除极值后的DataFrame
```
### 1.8.sigma_del(df, name, lst_n)
```python
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
```
### 1.9.percentile_del(df, name, lst_percentile)
```python
描述:
对某个日期的截面数据采用单期percentile法去极值

输入变量:
df:DataFrame
    name != None时第一列为N个股票代码(str),每行是n+个需要去极值的公司特征,shape = [N,1 + n]\
    name == None时每行为N个股票代码(str),shape = [1,N]

name:str,需要去极值的列名,name == None时是对单期面板去极值

lst_percentile:list:[percentile1(float,percentile方法左侧极值参数)\
    percentile2(float,percentile方法右侧极值参数)]\
    percentile方法百分位数为0到100的数

输出变量:
df:去除极值后的DataFrame
```
### 1.10.standardize_t(df_t,type)
```python
描述:
对相同日期不同股票或因子值的面板数据标准化或归一化

输入变量:
df_t:Series,某日期股票或因子值,shape = [N,]

type:str,type == 'standardize'时做标准化(z-score),type == 'normalize'时做归一化

输出变量:
df_standard_t:Series,某日期标准化后的股票或因子值,shape = [N,]

### 1.11.standardize_unstack(df_unstack,type = 'standardize'):
描述:
股票或因子值的面板数据标准化或归一化

输入变量:
df_unstack:面板DataFrame,index为时间,columns为股票代码或因子,shape = [T,N]

type:str,type == 'standardize'时做标准化(z-score),type == 'normalize'时做归一化

输出变量:
df_standard:面板DataFrame,标准化或归一化后的股票或因子值,index为时间,columns为股票代码或因子,shape = [T,N]
```
### 1.12.standardize_stack(df_stack,colname_lst,type = 'standardize')
```python
描述:
股票或因子值的堆栈数据标准化或归一化

输入变量:
df_stack:堆栈DataFrame,第一列是日期,第二列是股票代码或因子,其余n列是需要标准化的股票或因子值,shape = [T*N,2+n]

colname_lst:需要标准化或归一化的列名

type:str,type == 'standardize'时做标准化(z-score),type == 'normalize'时做归一化

输出变量:
df_standard:堆栈DataFrame,标准化或归一化后的股票或因子值,index为时间,columns为股票代码或因子,shape = [T,N]
```

## 2.factor singlesort
### 2.1.singlesort_id_t(df_t_stack,g)
```python
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
```
### 2.2.singlesort_ret_t(df_sort_t)
```python
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
```
### 2.3.singlesort_stack(df_stack,g,weighted,stated)
```python
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
```
### 2.4.singlesort_unstack(ret,character,g,weighted,stated,weight = None,state = None)
```python
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
```
### 2.5.long_short_cal(ret,long_only,fee = None,df_port = None)
```python
描述:根据分组收益率计算多空组合收益率

输入参数：
ret:DataFrame,分组收益率,index为datetime.date格式的日期,shape = [T,group]

long_only:bool,long_only == True时只计算多头收益,不计算多空收益,long_only == False时计算多空收益

fee:float,交易费率(一般用0.003),fee == None时不考虑交易费用

df_port:堆栈DataFrame,每个组合中各股票的权重,共4列,列名为['date','code','id','weight']
    date:datetime.date,日期
    code:str,股票代码
    id:float,1~g的整数,分组编号
    weight:float,归一化后的股票权重,相同日期和id的股票weight之和为1

输出参数：
df_long_short:DataFrame,index为datetime.date格式的日期,第一列为多空组合收益率,shape = [T,1]
```
### 2.6.net_val_cal(ret,show = False)
```python
描述:根据分组收益率计算各组累计净值或多空组合净值并画图

输入参数:
ret:DataFrame,分组收益率,index为datetime.date格式的日期,shape = [T,group]

show:bool,是否显示图像,当show == True时,会在本函数处暂停向下运行并显示分组净值图像；

show == False时不显示函数图像,只在主函数路径下导出cumulative net value.jpg的文件

输出参数：
cum_ret:DataFrame,分组累计净值,index为日期,第一列到最后一列为各个日期的分组净值,shape = [T,group]
```

## 3.factor test
### 3.1.ic_cal_stack(df_stack,ic_lst,type = 'rank')
```python
描述:
根据堆栈收益率和因子数据批量算因子normal ic和rank ic

输入变量:
df_stack:堆栈DataFrame,第一列为datetime.date类型的日期,第二列为str类型的股票代码,
    其它列是收益率及公司特征值,shape = [T*N,2+n]

ic_lst:list,计算涉及的变量名,ic_lst[0]为当期收益率列名,其余为相关因子列名

type:str,计算的IC类型,'normal'或'rank'

输出变量:
df_ic:面板DataFrame,index为日期,columns为因子名,values为各期因子ic
```
### 3.2.icir_cal(df_ic)
```python
描述:
根据因子IC序列计算IC均值和IR

输入变量:
df_ic:面板DataFrame,index为日期,columns为因子名,values为各期因子ic值,shape = [T,n]

输出变量:
icmean:Series,各因子ic均值,shape = [n],n == 1时type(icmean) = float

ir:Series,各因子ir,shape = [n],n == 1时type(ir) = float
```
### 3.3.ratios_cal(df_ratio_pre,multi)
```python
描述:
根据收益率和累计收益率序列计算年化夏普比率,最大回撤率

输入变量:
df_ratio_pre:面板DataFrame,index为日期,第一列为收益率,第二列为累计收益率,第三列为无风险利率

multi:计算年化夏普比率时的乘数,如日化转年化sqrt(252),月化转年化sqrt(12),周化转年化sqrt(52)

输出变量:
sharp,max_drawdown:float,年化夏普和最大回撤率
```
### 3.4.newey_west_test(ret,lag = None)
```python
描述：
计算收益率序列的均值,newey-west t值和p值

输入变量:
ret:array,收益率序列,shape = [T,1]

lag:int,Newey-West滞后阶数,默认为int(4*(T/100)^(2/9))

输出变量：
mean_ret:float,收益率均值

tval:float,收益率的Newey-west t值

pval:float,收益率的Newey-west p值
```
### 3.5.newey_west_reg(ret_factor,lag = None)
```python
描述：
将股票收益率或组合收益率向因子收益率回归,计算回归的alpha,beta,newey-west t值和p值

输入变量:
ret_factor:DataFrame,第一列为股票或组合收益率,第二列到最后一列为因子收益率,shape = [T,1 + factor_num]

lag:int,Newey-West滞后阶数,默认为int(4*(T/100)^(2/9))

输出变量：
param:Series,回归系数,当alpha_cal == True时计算包含截距,否则不包含截距

tval:Series,回归系数的Newey-west t值

pval:Series,回归系数的Newey-west p值
```
### 3.6.beta_rolling(df,p,rolling_w)
```python
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
```
### 3.7.fama_macbeth(df_stack,formula_lst,lag = None)
```python
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
```
