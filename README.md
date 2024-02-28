# 这是一个面向过程的因子回测框架
# 包含数据预处理，标准化，单分组，分析交易状态，构建多空组合，风险调整和FM回归(Barra)等功能，整体分为3部分：pro process,factor singlesort和factor test

## 1.pre_process
### 1.1.pretty_unstack(df_unstack_lst,format_lst)
'''python
描述:\
对一系列面板数据格式标准化,将日期格式标准化datetime.date并置为index,将各dataframe的日期对应一致

输入变量:\
df_unstack_lst:list,需要处理的面板数据列表,其中的每个元素是面板DataFrame,shape = [X,1]\
    DataFrame的第一列为str类型的日期,其它列是因子或公司特征值,shape = [T,1+N]

format_lst:list,各个面板数据对应的str类型的日期格式,如'%Y-%m-%d'等,其中每个元素为str

输出变量:\
df_pretty_lst:list,其中的每个元素是标准化后的面板DataFrame,shape = [X,1]\
    DataFrame的index为datetime.date类型的日期,index_name = 'date'\
    各列是因子或公司特征值,shape = [T,N]
'''

### 1.2.pretty_stack(df_stack,date_format)
描述:\
堆栈数据格式标准化,将日期格式标准化datetime.date,去除特征全为空值的日期

输入变量:\
df_stack:DataFrame,第一列为str类型的日期,第二列为str类型的股票代码,其它列是公司特征值,shape = [T*N,2+n]

date_format:str,日期格式,如'%Y-%m-%d'等

输出变量:\
df_pretty:DataFrame,第一列为datetime.date类型的日期,第二列str类型的股票代码,\
    columns = ['date','code',...],其余各列是公司特征值,shape = [T*N,2+n]

### 1.3.panels2stack(df_unstack_lst,colname_lst = None)
描述:\
将一系列格式标准化后的面板数据(经过pretty_unstack处理后)合并并转化为堆栈数据

输入变量:\
df_unstack_lst:list,其中的每个元素是标准化后的面板DataFrame,shape = [X,1]\
    DataFrame的index为datetime.date类型的日期,index_name = 'date'\
    各列是因子或公司特征值,shape = [T,N]

colname_lst:输出的堆栈数据列名,colname_lst = None时不设置列名

输出变量:\
df:重新设置index后的堆栈数据,其第一列为date,第二列为code

### 1.4.stack2panels(df_stack,idxname,colname,panelname_lst)
描述:\
将格式标准化后的堆栈数据(经过pretty_stack处理后)批量转化为一系列面板数据

输入变量:\
df_stack:堆栈DataFrame,index为普通index(1,2,3...),列名中包含idxname,colname,panelname_lst

idxname:生成的面板DataFrame的index名

colname:生成的面板DataFrame的column名

panelname_lst:原堆栈数据中需要做反堆栈的列名

输出变量:\
df_unstack_lst:list,其中每个元素是面板DataFrame,shape = [len(panelname_lst),1],\
    DataFrame的index名为idxname,column名为colname,shape = [len(colname),len(idxname)],\
    值为原数据以panelname_lst中值为列名的值

### 1.5.del_outlier_unstack(df_unstack,method,arg)
描述：\
对面板数据进行去极值,将超出范围的数据改为临界点

输入变量：\
df_unstack:面板DataFrame,index为datetime.date类型的日期,各列是因子或公司特征值,shape = [T,N]

method:str,去极值方式,共3种:{'mad','sigma','percentile'}\
    mad:根据数据中位数去极值\
    sigma:根据数据方差去极值\
    percentile:根据数据分位点去极值

arg:list,根据method确定的参数\
    method == mad:arg = [n1(mad方法左侧极值参数),n2(mad方法右侧极值参数)]\
    method == sigma:arg = [n1(sigma方法左侧极值参数),n2(sigma方法右侧极值参数)]\
    method == percentile:arg = [percentile1(float,不带百分号的percentile方法百分位数,左侧极值参数)\
    percentile2(float,不带百分号的percentile方法百分位数,右侧极值参数)]

输出变量:\
df_del:面板DataFrame,去极值后的数据

### 1.6.del_outlier_stack(df_stack,name,method,arg)
描述：\
对堆栈的DataFrame进行去极值,将超出范围的数据改为临界点

输入变量：\
df_stack:堆栈的DataFrame,第一列为日期(datetime.date),第二列为股票代码(str),shape = [T*N,2 + n]

name:str,需要去极值的列名

method:str,去极值方式,共3种:{'mad','sigma','percentile'}\
    mad:根据数据中位数去极值\
    sigma:根据数据方差去极值\
    percentile:根据数据分位点去极值

arg:list,根据method确定的参数\
    method == mad:arg = [n1(mad方法左侧极值参数),n2(mad方法右侧极值参数)]\
    method == sigma:arg = [n1(sigma方法左侧极值参数),n2(sigma方法右侧极值参数)]\
    method == percentile:arg = [percentile1(float,不带百分号的percentile方法百分位数,左侧极值参数)\
    percentile2(float,不带百分号的percentile方法百分位数,右侧极值参数)]

输出变量:
df_del:DataFrame,去极值后的数据

### 1.7.mad_del(df, name, lst_n)
描述:\
对某个日期的截面数据采用单期MAD法去极值

输入变量:\
df:DataFrame\
    name != None时第一列为N个股票代码(str),每行是n+个需要去极值的公司特征,shape = [N,1 + n]\
    name == None时每行为N个股票代码(str),shape = [1,N]

name:str,需要去极值的列名,name == None时是对单期面板去极值

lst_n:list:[n1(mad方法左侧极值参数),n2(mad方法右侧极值参数)]\
    mad方法极值参数为中位数设置位置,阈值设置在几个绝对偏差中位数上

输出变量:\
df:去除极值后的DataFrame

### 1.8.sigma_del(df, name, lst_n)
描述:\
对某个日期的截面数据采用单期sigma法去极值

输入变量:\
df:DataFrame\
    name != None时第一列为N个股票代码(str),每行是n+个需要去极值的公司特征,shape = [N,1 + n]\
    name == None时每行为N个股票代码(str),shape = [1,N]

name:str,需要去极值的列名,name == None时是对单期面板去极值

lst_n:list:[n1(sigma方法左侧极值参数),n2(sigma方法右侧极值参数)]\
    sigma方法极值参数为标准差设置位置,阈值设置在几个标准差的地方,一般为3个标准差(n1=n2=3)

输出变量:\
df:去除极值后的DataFrame

### 1.9.percentile_del(df, name, lst_percentile)
描述:\
对某个日期的截面数据采用单期percentile法去极值

输入变量:\
df:DataFrame\
    name != None时第一列为N个股票代码(str),每行是n+个需要去极值的公司特征,shape = [N,1 + n]\
    name == None时每行为N个股票代码(str),shape = [1,N]

name:str,需要去极值的列名,name == None时是对单期面板去极值

lst_percentile:list:[percentile1(float,percentile方法左侧极值参数)\
    percentile2(float,percentile方法右侧极值参数)]\
    percentile方法百分位数为0到100的数

输出变量:\
df:去除极值后的DataFrame

### 1.10.standardize_t(df_t,type):
描述:\
对相同日期不同股票或因子值的面板数据标准化或归一化

输入变量:\
df_t:Series,某日期股票或因子值,shape = [N,]

type:str,type == 'standardize'时做标准化(z-score),type == 'normalize'时做归一化

输出变量:\
df_standard_t:Series,某日期标准化后的股票或因子值,shape = [N,]

### 1.11.standardize_unstack(df_unstack,type = 'standardize'):
描述:\
股票或因子值的面板数据标准化或归一化

输入变量:\
df_unstack:面板DataFrame,index为时间,columns为股票代码或因子,shape = [T,N]

type:str,type == 'standardize'时做标准化(z-score),type == 'normalize'时做归一化

输出变量:\
df_standard:面板DataFrame,标准化或归一化后的股票或因子值,index为时间,columns为股票代码或因子,shape = [T,N]

### 1.12.standardize_stack(df_stack,colname_lst,type = 'standardize'):
描述:\
股票或因子值的堆栈数据标准化或归一化

输入变量:\
df_stack:堆栈DataFrame,第一列是日期,第二列是股票代码或因子,其余n列是需要标准化的股票或因子值,shape = [T*N,2+n]

colname_lst:需要标准化或归一化的列名

type:str,type == 'standardize'时做标准化(z-score),type == 'normalize'时做归一化

输出变量:\
df_standard:堆栈DataFrame,标准化或归一化后的股票或因子值,index为时间,columns为股票代码或因子,shape = [T,N]
