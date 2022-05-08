import os
import re
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

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


def modelSVM(category_names,test_labels,test_contents,train_labels,train_contents):
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
    #result = pipeline_svm.predict(test_contents)
    #classification_report生成报告
    #report = classification_report(test_labels, result)
    #report = classification_report(test_labels, result, target_names=category_names)


    warnings.filterwarnings('ignore')

    accuracy = cross_val_score(pipeline_svm, test_contents,test_labels, scoring='accuracy', cv=5)
    f1_micro = cross_val_score(pipeline_svm, test_contents,test_labels,  scoring = 'f1_micro', cv=5)
    f1_measure = cross_val_score(pipeline_svm, test_contents, test_labels, scoring='f1', cv=5)
    precision = cross_val_score(pipeline_svm, test_contents, test_labels, scoring='average_precision', cv=5)
    recall = cross_val_score(pipeline_svm,test_contents,test_labels, scoring='recall', cv=5)

    print("SVM准确率:", accuracy.mean())
    print("SVMf1_micro:", f1_micro.mean())
    print("SVMf1_measure:", f1_measure.mean())
    print("SVM精确率:", precision.mean())
    print("SVM召回率:", recall.mean())


def modelBayes(category_names,test_labels,test_contents,train_labels,train_contents):
    # 实例化
    tf = TfidfVectorizer()
    # 将训练集中的新闻文本数据进行特征抽取,返回一个sparse矩阵
    x_train = tf.fit_transform(train_contents)
    # 将测试集中的新闻文本数据进行特征抽取，返回一个sparse矩阵
    x_test = tf.transform(test_contents)

    # alpha为拉普拉斯修正的α
    bayes = MultinomialNB(alpha=1.0)
    bayes.fit(x_train, train_labels)

    y_predict = bayes.predict(x_test)

    accuracy = cross_val_score(bayes,x_test, test_labels, scoring='accuracy', cv=5)
    f1_micro = cross_val_score(bayes, x_test, test_labels, scoring='f1_micro', cv=5)
    f1_measure = cross_val_score(bayes, x_test, test_labels, scoring='f1', cv=5)
    precision = cross_val_score(bayes, x_test, test_labels, scoring='average_precision', cv=5)
    recall = cross_val_score(bayes,x_test, test_labels, scoring='recall', cv=5)
    f1 = cross_val_score(bayes, x_test, test_labels, scoring='f1', cv=5)

    print("朴素贝叶斯准确率:", accuracy.mean())
    print("朴素贝叶斯f1_micro:", f1_micro.mean())
    print("朴素贝叶斯f1_measure:", f1_measure.mean())
    print("朴素贝叶斯精确率:", precision.mean())
    print("朴素贝叶斯召回率:", recall.mean())
    print("朴素贝叶斯f1:", f1.mean())



def modelLogisticRegression(category_names,test_labels,test_contents,train_labels,train_contents):
    #同上
    pipeline_log= Pipeline([('tfidf', TfidfVectorizer(max_features=10000)),
                         ('clf', LogisticRegression(max_iter=10))])

    pipeline_log.fit(train_contents, train_labels)
    result =  pipeline_log.predict(test_contents)

    # classification_report生成报告
    # report = classification_report(test_labels, result)
    # report = classification_report(test_labels, result, target_names=category_names)

    warnings.filterwarnings('ignore')

    accuracy = cross_val_score( pipeline_log, test_contents, test_labels, scoring='accuracy', cv=5)
    f1_micro = cross_val_score( pipeline_log,test_contents, test_labels, scoring='f1_micro', cv=5)
    f1_measure = cross_val_score(pipeline_log, test_contents, test_labels, scoring='f1', cv=5)
    precision = cross_val_score( pipeline_log, test_contents, test_labels, scoring='average_precision', cv=5)
    recall = cross_val_score( pipeline_log, test_contents, test_labels, scoring='recall', cv=5)

    print("逻辑斯蒂回归准确率:", accuracy.mean())
    print("逻辑斯蒂回归f1_micro:", f1_micro.mean())
    print("逻辑斯蒂回归f1_measure:", f1_measure.mean())
    print("逻辑斯蒂回归精确率:", precision.mean())
    print("逻辑斯蒂回归召回率:", recall.mean())



if __name__ == '__main__':
    category_dir, test_labels, test_contents, train_labels, train_contents=dataWashing()
    modelSVM(category_dir,test_labels,test_contents,train_labels,train_contents)
    modelBayes(category_dir,test_labels,test_contents,train_labels,train_contents)
    modelLogisticRegression(category_dir, test_labels, test_contents, train_labels, train_contents)