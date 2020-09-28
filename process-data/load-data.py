import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
plt.style.use('ggplot') 
import os
file_path = os.path.dirname(__file__)
data_path = os.path.join(file_path, "../data")
# print(os.listdir(data_path))


item = pd.read_csv(os.path.join(data_path, 'items.csv'))
item_category = pd.read_csv(os.path.join(data_path, 'item_categories.csv'))
train = pd.read_csv(os.path.join(data_path, 'sales_train.csv'))
test = pd.read_csv(os.path.join(data_path, 'test.csv'))
shop = pd.read_csv(os.path.join(data_path, 'shops.csv'))

# summary data
def eda(data):
    print("----------Top-5- Record----------")
    print(data.head(5))
    print("-----------Information-----------")
    print(data.info())
    print("-----------Data Types-----------")
    print(data.dtypes)
    print("----------Missing value-----------")
    print(data.isnull().sum())
    print("----------Null value-----------")
    print(data.isna().sum())
    print("----------Shape of Data----------")
    print(data.shape)


def graph_insight(data):
    print(set(data.dtypes.tolist()))
    df_num = data.select_dtypes(include = ['float64', 'int64'])
    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);
    plt.plot(df_num)
    plt.show()

#graph insight for items data
def graph_insight_items(data):
    print(set(data.dtypes.tolist()))
    df_num = data.select_dtypes(include = ['float64', 'int64'])
    plt.hist(df_num['item_category_id'], bins=50, alpha=0.5, histtype='stepfilled', color='blue', edgecolor='none')
    plt.xlabel('item_category_id');
    plt.ylabel('number of items');
    plt.show()
    print("draw done!")
#graph insight for train data
def graph_insight_train(data):
    print(set(data.dtypes.tolist()))
    df_num = data.select_dtypes(include = ['float64', 'int64'])

    #1. graph of total transactions per month
    # plt.hist(df_num['date_block_num'],bins= 32, alpha=0.5, histtype='stepfilled', color='blue', edgecolor='none')
    # plt.xlabel('date_block_num');
    # plt.ylabel('number of transactions');

    #2. graph of total transactions per shop
    # plt.hist(df_num['shop_id'],bins= 80, alpha=0.5, histtype='stepfilled', color='blue', edgecolor='none')
    # plt.xlabel('shop_id');
    # plt.ylabel('number of transactions');

    #3. graph of total transactions per item_id
    plt.hist(df_num['item_id'],bins=100, alpha=0.5, histtype='stepfilled', color='blue', edgecolor='none')
    plt.xlabel('item_id');
    plt.ylabel('number of transactions');

    #4.1. remove outliner and draw a graph of total transactions per price
    # print(df_num.shape)
    # df_num = df_num.drop(df_num[(df_num.item_price < 0) & (df_num.item_price > 15000)].index)
    # print(df_num.shape)
    # plt.hist(df_num['item_price'],bins= 100, alpha=0.5, histtype='stepfilled', color='blue', edgecolor='none')
    # plt.xlabel('item_price');
    # plt.ylabel('number of transactions');


    plt.show()
    print("draw done!")

def drop_duplicate(data, subset):
    print('Before drop shape:', data.shape)
    before = data.shape[0]
    data.drop_duplicates(subset,keep='first', inplace=True)
    data.reset_index(drop=True, inplace=True)
    print('After drop shape:', data.shape)
    after = data.shape[0]
    print('Total Duplicate:', before-after)

#1 Sale train data
eda(train)
graph_insight_train(train)
subset = ['date', 'date_block_num', 'shop_id', 'item_id','item_cnt_day']
drop_duplicate(train, subset = subset)

#2 Test data
# eda(test)
# graph_insight(test)

#3 Items data
# eda(item)
# graph_insight_items(item)

#4 Item_categories data
# eda(item_category)
# graph_insight(item_category)

#5 shops data
# eda(shop)
# graph_insight(shop)


#Statistic number of product sold per day
print(train['item_cnt_day'].describe())
