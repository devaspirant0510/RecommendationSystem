import mlxtend
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# 사용자 물품 구매 데이터
data = np.array([["콜라","과자","라면"],
                 ["라면","햇반"],
                 ["콜라","라면","햇반"],
                 ["라면","과자","햇반","콜라"]],dtype=object)

# 희소 행렬 Sparse Matrix
te = TransactionEncoder()
te_SM = te.fit(data).transform(data)
print(te_SM)

df = pd.DataFrame(te_SM,columns=te.columns_)
print(df)
print(apriori(df,min_support=0.5,use_colnames=True))

