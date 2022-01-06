import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

# ensures that the output is on a single line
pd.set_option('display.expand_frame_repr', False)

from mlxtend.frequent_patterns import apriori, association_rules

############################################
# 1.Perform data preprocessing
############################################
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()

# POST: It means postage information.It would be better to take it out and move on
df[df["StockCode"] == "POST"]["Description"].unique()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_data_prep(dataframe):
    dataframe.drop(dataframe[dataframe["StockCode"] == "POST"].index, inplace=True)
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


df = retail_data_prep(df)

df_deu = df[df["Country"] == "Germany"]
df_deu.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).head()
df_deu.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

df_deu.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x>0 else 0).head()

############################################
# 2.Produce association rules through Germany customers
############################################

def create_invoice_product_df(dataframe, id = False):
    if id:
        return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x : 1 if x>0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)


deu_inv_pro_df = create_invoice_product_df(df_deu, id = True)
deu_inv_pro_df.head()

frequent_itemsets = apriori(deu_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head()

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()
rules.sort_values("lift", ascending=False).head()
############################################
# 3.What are the names of the products whose IDs are given?
# User 1 product id: 21987
# User 2 product id: 23235
# User 3 product id: 22747
############################################
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


product_id1 = 21987
product_id2 = 23235
product_id3 = 22747
check_id(df, product_id1)
check_id(df, product_id2)
check_id(df, product_id3)
############################################
# 4.Make a product recommendation for the users in the cart
# What are the names of the recommended products?
############################################
def arl_recommender(rules_df, product_id, rec_count = 1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommend_list = []

    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommend_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommend_list = list({item for item_list in recommend_list for item in item_list})

    [check_id(df, item) for item in recommend_list[:rec_count]]

    return recommend_list[:rec_count]


arl_recommender(rules, product_id1, 3)
arl_recommender(rules, product_id2, 3)
arl_recommender(rules, product_id3, 3)