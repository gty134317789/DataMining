import os
import re

# replace any character that is not digit or letter or space with empty string
replace_with_empty_pattern = re.compile(r'[^A-Za-z0-9\s]')
# replace consecutive spaces and enters(\n) with a single space
replace_with_single_space_pattern = re.compile(r'\s{2,}|[^\S ]')


def preprocess_content(content):
    return re.sub(
        replace_with_single_space_pattern, ' ',
        re.sub(replace_with_empty_pattern, '', content)
    )

#引入数据集
directory = './data/20news-18828'
#切割目录，os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
category_names = os.listdir(directory)
#news_contents作为X，列表
news_contents = list()
#news_labels作为Y，列表
news_labels = list()
#将二者结合为目录
for i in range(len(category_names)):
	category = category_names[i]
    #category_dir就是结合后的目录
	category_dir = os.path.join(directory, category)
    #此处file_path就是每一个文件的对应路径
#print(category_dir)
for file_name in os.listdir(category_dir):
    file_path = os.path.join(category_dir, file_name)
    #raw_content获取内容
    #strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    raw_content = open(file_path, encoding='latin1').read().strip()
    #print(file_path)
    #print(raw_content)
    #news_content是新闻的内容
    news_content = preprocess_content(raw_content)
    #更新news_labels，作为编号
    news_labels.append(i + 1)
    # 更新news_contents，作为内容
    news_contents.append(news_content)




