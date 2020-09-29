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

#summary item_cnt_day data
def remove_item_cnt_day_outlier(train):
    print(train['item_cnt_day'].sort_values(ascending=False).head(10))
    print(train['item_cnt_day'].describe())
    print(train[train.item_cnt_day>500])
    #draw item_cnt_day graph
    plt.figure(figsize=(10, 5))
    color = sns.color_palette("hls", 8)
    plt.xlim(-200, 3500)
    sns.boxplot(x=train.item_cnt_day, color="red", palette="Set3")
    plt.show()
    #remove outliers > 1000
    train = train[train.item_cnt_day<501]
    #draw item_cnt_day graph again
    plt.figure(figsize=(10, 5))
    color = sns.color_palette("hls", 8)
    plt.xlim(-200, 3500)
    sns.boxplot(x=train.item_cnt_day, color="red", palette="Set3")
    plt.show()

#summary item_price data
def remove_item_price_outlier(train):
    print(train['item_price'].sort_values(ascending=False).head(10))
    print(train['item_price'].describe())
    #draw item_cnt_day graph
    plt.figure(figsize=(10, 5))
    color = sns.color_palette("hls", 8)
    plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
    sns.boxplot(x=train.item_price, color="red", palette="Set3")
    plt.show()

    #remove outliers > 1000
    print(train[train.item_price>=50000])
    train = train[train.item_price<50000]
    #draw item_cnt_day graph again
    plt.figure(figsize=(10, 5))
    color = sns.color_palette("hls", 8)
    plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
    sns.boxplot(x=train.item_price, color="red", palette="Set3")
    plt.show()

# remove_item_cnt_day_outlier(train)
# remove_item_price_outlier(train)

#process shop_name -> shop city | shop type | shop name
#summary 
print(shop.head(10))
shop['shop_name'] = shop['shop_name'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+','').str.strip()
shop['shop_city'] = shop['shop_name'].str.partition(' ')[0]
shop['shop_type'] = shop['shop_name'].apply(lambda x: 'мтрц' if 'мтрц' in x else 'трц' if 'трц' in x else 'трк' if 'трк' in x else 'тц' if 'тц' in x else 'тк' if 'тк' in x else 'NO_DATA')
print(shop.head(10))
