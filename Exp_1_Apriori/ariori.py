import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

#首先整理下数据集
def loadDataSet():
    # dataSet = [['面包', '可乐', '麦片'],
    #        2 X ['牛奶', '可乐'],
    #            ['面包', '鸡蛋', '麦片'],
    #            ['牛奶', '面包', '面包'],
    #        2 X ['牛奶', '面包', '可乐'],
    #            ['面包', '可乐']
    #            ['牛奶','面包','鸡蛋','麦片']
    # ]
    return [[1, 2, 3], [4, 2], [4, 1, 3], [4, 2], [1, 5, 3], [4, 1, 2], [4, 1, 5, 3], [4, 1, 2], [1, 2]]

# 获取候选1项集，dataSet为事务集。返回一个list，每个元素都是set集合
def createC1(dataSet):
    C1 = []  # 元素个数为1的项集（非频繁项集，因为还没有同最小支持度比较）
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()  # 这里排序是为了，生成新的候选集时可以直接认为两个n项候选集前面的部分相同
    # 因为除了候选1项集外其他的候选n项集都是以二维列表的形式存在，所以要将候选1项集的每一个元素都转化为一个单独的集合。
    return list(map(frozenset, C1))  # map(frozenset, C1)的语义是将C1由Python列表转换为不变集合（frozenset，Python中的数据结构）

# 找出候选集中的频繁项集
# dataSet为全部数据集，Ck为大小为k（包含k个元素）的候选项集，minSupport为设定的最小支持度
def scanD(dataSet, Ck, minSupport):
    ssCnt = {}  # 记录每个候选项的个数
    for tid in dataSet:
        for can in Ck:
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can, 0) + 1  # 计算每一个项集出现的频率
    numItems = float(len(dataSet))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)  # 将频繁项集插入返回列表的首部
        supportData[key] = support
    return retList, supportData  # retList为在Ck中找出的频繁项集（支持度大于minSupport的），supportData记录各频繁项集的支持度

# 通过频繁项集列表Lk和项集个数k生成候选项集C(k+1)。
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 前k-1项相同时，才将两个集合合并，合并后才能生成k+1项
            L1 = list(Lk[i])[:k - 2];
            L2 = list(Lk[j])[:k - 2]  # 取出两个集合的前k-1个元素
            L1.sort();
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

# 获取事务集中的所有的频繁项集
# Ck表示项数为k的候选项集，最初的C1通过createC1()函数生成。Lk表示项数为k的频繁项集，supK为其支持度，Lk和supK由scanD()函数通过Ck计算而来。
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)  # 从事务集中获取候选1项集
    D = list(map(set, dataSet))  # 将事务集的每个元素转化为集合
    L1, supportData = scanD(D, C1, minSupport)  # 获取频繁1项集和对应的支持度
    L = [L1]  # L用来存储所有的频繁项集
    k = 2
    while (len(L[k - 2]) > 0):  # 一直迭代到项集数目过大而在事务集中不存在这种n项集
        Ck = aprioriGen(L[k - 2], k)  # 根据频繁项集生成新的候选项集。Ck表示项数为k的候选项集
        Lk, supK = scanD(D, Ck, minSupport)  # Lk表示项数为k的频繁项集，supK为其支持度
        L.append(Lk);
        supportData.update(supK)  # 添加新频繁项集和他们的支持度
        k += 1
    return L, supportData

#使用seaborn画热力图
def draw_seaborn(L, suppData, n):
    data = []
    for i in range(n):
        list1 = []
        list1.append(L[1][i])
        list1.append(suppData[L[1][i]] * 9)
        list1.append(suppData[L[1][i]])
        data.append(list1)
    # print(data)
    df = pd.DataFrame(data, columns=['item', 'count', 'support'])
    # print(df)
    df = df.pivot(index='item', columns='count', values='support')  # 将一个dataframe的记录数据整合成表格(类似透视表）
    # print(df)
    heapmap = sns.heatmap(df, fmt='d', linewidths=.5, cmap="OrRd")  # fmt-显示数字格式 linewidths-热力图矩阵之间的间隔大小 camp-绿到蓝
    plt.tight_layout()  # 解决未全部显示的问题
    plt.show()
    return sns.heatmap(df, fmt='d', linewidths=.5)

def d_deal(L, sData):
    min = -1
    DataList = []
    for i in range(len(L[1])):
        for j in L[1][i]:
            if i > min:
                min = i
                List = []
                List.append(sData[L[1][i]])
            List.append(j)
        DataList.append(List)
    num = {}
    for m in range(len(L[0])):
        num[len(L[0]) - m] = L[0][m]
    return DataList, num

#计算2阶频繁项集的置信度
def confidence(DataList, Datanum, suppData):
    c_data = {}
    for num in range(len(DataList)):
        s1 = ''
        s2 = ''
        s1 = str(DataList[num][1]) + '->' + str(DataList[num][2])
        c_data[s1] = DataList[num][0] / suppData[Datanum[DataList[num][1]]]
        s2 = str(DataList[num][2]) + '->' + str(DataList[num][1])
        c_data[s2] = DataList[num][0] / suppData[Datanum[DataList[num][2]]]
    return c_data

#计算提升度
def Lift(DataList, c_data, Datanum, suppData):
    L_data = {}
    for num in range(len(DataList)):
        s1 = ''
        s2 = ''
        s1 = str(DataList[num][1]) + '->' + str(DataList[num][2])
        L_data[s1] = c_data[s1] / suppData[Datanum[DataList[num][2]]]
        s2 = str(DataList[num][2]) + '->' + str(DataList[num][1])
        L_data[s2] = c_data[s2] / suppData[Datanum[DataList[num][1]]]
    return L_data

#定义最小置信度阈值
def limit(c_data, L_data, minconfidence=0.5):
    limit_data = []
    for k in c_data.keys():
        data = []
        if (c_data[k] >= minconfidence):
            data.append(k)
            data.append(c_data[k])
            data.append(L_data[k])
            limit_data.append(data)
    return limit_data

#主函数
if __name__ == '__main__':
    dataSet = loadDataSet()  # 获取事务集。每个元素都是列表
    L, suppData = apriori(dataSet, minSupport=0.2)
    print(L[1])
    draw_seaborn(L, suppData, len(L[1]))
    DataList, Datanum = d_deal(L, suppData)
    confidence_data = confidence(DataList, Datanum, suppData)
    Lift_data = Lift(DataList, confidence_data, Datanum, suppData)
    limit_data = limit(confidence_data, Lift_data, minconfidence=0.5)
    print('\n')
    for i in limit_data:
        print(i)
