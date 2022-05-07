"""
习惯用面向对象的思维了,
所以可能函数写的比较多...
其实这边用面向过程的思维更好
"""
import pandas as pd

#将文件转换为csv格式
def transform_data():
    df=pd.read_excel('数据挖掘与分析-0217191-你说对就队打分统计.xlsx')
    df.to_csv('原文件的csv形式.csv',encoding='utf_8_sig',index=False)

#获取转换后的文件
def get_data():
    file=pd.read_csv('原文件的csv形式.csv')
    return file

#清洗数据,删除空值
def drop_nulldata(file):
    new_File=file.dropna()   #清洗空值
    return(new_File)

#将标题数据统一
def standardize_data(new_File):
    new_File.loc[:,"Unnamed: 6"] = "你说对就对"    #统一标题
    new_File.to_csv('数据挖掘与分析-0217191-你说对就队打分统计.csv', encoding='utf_8_sig', index=False)

#主函数
if __name__ == '__main__':
    transform_data()
    file=get_data()
    new_file=drop_nulldata(file)
    standardize_data(new_file)


