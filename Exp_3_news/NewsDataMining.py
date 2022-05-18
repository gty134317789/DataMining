import os
import re
import warnings


from matplotlib import pyplot as plt
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, RepeatedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#定义两条正则表达式
#用空字符串替换任何非数字、字母或空格的字符
replace_with_empty_pattern = re.compile(r'[^A-Za-z0-9\s]')
#用单个空格输入换连续空格
replace_with_single_space_pattern = re.compile(r'\s{2,}|[^\S ]')

#定义数据清洗的函数，利用正则表达式清洗数据，去除空格和标点
def preprocess_content(content):
    return re.sub(
        replace_with_single_space_pattern, ' ',
        re.sub(replace_with_empty_pattern, '', content)
    )

#数据清洗主函数
def dataWashing():
    # 引入数据集
    directory = './data/20news-18828'
    # 切割目录，os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    category_names = os.listdir(directory)
    # news_contents作为X，列表
    news_contents = list()
    # news_labels作为Y，列表
    news_labels = list()
    # 将二者结合为目录
    for i in range(len(category_names)):
        category = category_names[i]
        # category_dir就是结合后的目录
        category_dir = os.path.join(directory, category)
        # 此处file_path就是每一个文件的对应路径
        # print(category_dir)
        for file_name in os.listdir(category_dir):
            file_path = os.path.join(category_dir, file_name)
            # raw_content获取内容
            # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
            raw_content = open(file_path, encoding='latin1').read().strip()
            # print(file_path)
            # print(raw_content)
            # news_content是新闻的内容
            news_content = preprocess_content(raw_content)
            # 更新news_labels，作为编号，list类型
            news_labels.append(i + 1)
            # print(news_labels)
            # 更新news_contents，作为内容,list类型
            news_contents.append(news_content)

    # 对文本进行分类，8：2
    # train_X,Y是训练集，test_X,Y是测试集
    train_contents, test_contents, train_labels, test_labels = \
        train_test_split(news_contents, news_labels, shuffle=True, test_size=0.2)
    return (category_dir, test_labels, test_contents, train_labels, train_contents)

#SVM主函数
def modelSVM(category_names, test_labels, test_contents, train_labels, train_contents, result_list=None):
    # 将文本文档转化为TF-IDF向量
    tfidf_vectorizer = TfidfVectorizer()
    # LinearSVC：线性支持向量分类器
    svm_classifier = LinearSVC(verbose=True)
    # 选10000个词语作为文档特征词
    chi2_feature_selector = SelectKBest(chi2, k=10000)
    # pipeline流水线：向量化（vectorizer） => 转换器（transformer） => 分类器（classifier）
    pipeline_svm = Pipeline(memory=None, steps=[
        ('tfidf', tfidf_vectorizer),
        ('chi2', chi2_feature_selector),
        ('svm', svm_classifier),
    ])

    pipeline_svm.fit(train_contents, train_labels)
    result = pipeline_svm.predict(test_contents)
    #classification_report生成报告
    report = classification_report(test_labels, result)
    #report = classification_report(test_labels, result, target_names=category_names)

    accuracy=accuracy_score(test_labels, result)
    f1 = f1_score(test_labels, result, average='macro')
    recall = recall_score(test_labels,result, average='macro')
    precision = precision_score(test_labels, result, average='macro')
    predY = cross_val_predict(pipeline_svm, test_contents, test_labels, cv=5)
    warnings.filterwarnings('ignore')

    print("----------------------------------------------------")
    print("SVM准确率:", accuracy)
    print("SVMf1_measure:", f1)
    print("SVM精确率:", precision)
    print("SVM召回率:", recall)
    print(predY)
    svm_list=[accuracy,f1,precision,recall]
    return svm_list

#朴素贝叶斯主函数
def modelBayes(category_names,test_labels,test_contents,train_labels,train_contents):
    tfidf_vectorizer = TfidfVectorizer()
    bayes = BernoulliNB()
    chi2_feature_selector = SelectKBest(chi2, k=10000)
    pipeline = Pipeline(memory=None, steps=[
        ('tfidf', tfidf_vectorizer),
        ('chi2', chi2_feature_selector),
        ('bayes', bayes),
    ])

    pipeline.fit(train_contents, train_labels)
    result = cross_val_predict(pipeline, test_contents, test_labels, cv=5)

    accuracy = accuracy_score(test_labels, result)
    f1 = f1_score(test_labels, result, average='macro')
    recall = recall_score(test_labels, result, average='macro')
    precision = precision_score(test_labels, result, average='macro')
    predY = cross_val_predict(pipeline, test_contents, test_labels, cv=5)
    warnings.filterwarnings('ignore')

    print("----------------------------------------------------")
    print("朴素贝叶斯准确率:", accuracy)
    print("朴素贝叶斯f1_measure:", f1)
    print("朴素贝叶斯精确率:", precision)
    print("朴素贝叶斯召回率:", recall)
    print(predY)
    bayes_list = [accuracy, f1, precision, recall]
    return bayes_list

#逻辑斯蒂回归主函数
def modelLogisticRegression(category_names,test_labels,test_contents,train_labels,train_contents):
    #同上
    pipeline_log= Pipeline([('tfidf', TfidfVectorizer(max_features=10000)),
                         ('clf', LogisticRegression(max_iter=100))])

    pipeline_log.fit(train_contents, train_labels)
    result =  pipeline_log.predict(test_contents)


    warnings.filterwarnings('ignore')

    accuracy = accuracy_score(test_labels, result)
    f1 = f1_score(test_labels, result, average='macro')
    recall = recall_score(test_labels, result, average='macro')
    precision = precision_score(test_labels, result, average='macro')
    predY = cross_val_predict(pipeline_log, test_contents, test_labels, cv=5)
    warnings.filterwarnings('ignore')

    print("----------------------------------------------------")
    print("逻辑斯蒂回归准确率:", accuracy)
    print("逻辑斯蒂回归f1_measure:", f1)
    print("逻辑斯蒂回归精确率:", precision)
    print("逻辑斯蒂回归召回率:", recall)
    print(predY)
    print("----------------------------------------------------")
    logisticregression_list = [accuracy, f1, precision, recall]
    return logisticregression_list

#绘制柱状图
def draw_column(svm_list, bayes_list,logisticregression_list):
    # 定义列表，用于绘图
    name_list = ["accuracy", "f1_measure", "precision", "recall"]
    x = list(range(len(svm_list)))  # x=[0,1]，为生成柱的x坐标
    total_width, n = 0.7, 3
    width = total_width / n  # 每个柱状图的宽度
    # 绘制svm模型柱状图
    plt.bar(x, svm_list, width=width, label="svm模型", tick_label=name_list)
    for a, b in zip(x, svm_list):  # 柱子上的数字显示
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=10)
    # 绘制贝叶斯模型柱状图
    for i in range(len(x)):  # 完成一组柱状图的绘制后，增加x坐标，以避免柱状图重叠
        x[i] = x[i] + width
    plt.bar(x, bayes_list, width=width, label="贝叶斯模型", tick_label=name_list)
    for a, b in zip(x, bayes_list):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=10)
    # 绘制贝叶斯模型柱状图
    for i in range(len(x)):  # 完成一组柱状图的绘制后，增加x坐标，以避免柱状图重叠
        x[i] = x[i] + width
    plt.bar(x, logisticregression_list, width=width, label="逻辑斯蒂回归模型", tick_label=name_list)
    for a, b in zip(x, logisticregression_list):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=10)

    # 设计对比图文字
    plt.ylabel("指标数值", fontsize=10)
    plt.title("三个模型指标对比", fontsize=15)
    plt.legend(loc=(0.93, 0.95), frameon=False, fontsize=10)  # 调整图例位置
    plt.show()


if __name__ == '__main__':
    category_dir, test_labels, test_contents, train_labels, train_contents=dataWashing()
    svm_list=modelSVM(category_dir,test_labels,test_contents,train_labels,train_contents)
    bayes_list=modelBayes(category_dir,test_labels,test_contents,train_labels,train_contents)
    logisticregression_list=modelLogisticRegression(category_dir, test_labels, test_contents, train_labels, train_contents)
    draw_column(svm_list,bayes_list,logisticregression_list)

