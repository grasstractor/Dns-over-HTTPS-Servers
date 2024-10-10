import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder
import joblib  # 用于保存编码器
import numpy as np

# 从CSV文件读取数据
print("读取CSV文件...")
df = pd.read_csv('../探测的原始数据/final_extracted_data.csv')
print("CSV文件读取完成")

# 样本去重，根据IP地址以外的所有列去重
print("去重处理...")
df = df.drop_duplicates(subset=df.columns.difference(['IP']))
df.reset_index(drop=True, inplace=True)  # 重置索引
print("去重处理完成")

# 处理 Server 列
print("处理 Server 列...")
df['Server'] = df['Server'].fillna('Unknown Server')
hasher = FeatureHasher(input_type='string', n_features=64)
server_hashed = hasher.transform(df['Server'].apply(lambda x: [x])).toarray()
server_hashed_df = pd.DataFrame(server_hashed, columns=[f'Server_{i}' for i in range(64)])
print("Server 列处理完成，维度：", server_hashed_df.shape)

# 保存 FeatureHasher 编码器
joblib.dump(hasher, './编码器/feature_hasher_server.pkl')
print("Server 编码器已保存")

# 处理 Cache-Control 和 Content-Type 列
print("处理 Cache-Control 和 Content-Type 列...")
df['Cache-Control'] = df['Cache-Control'].fillna('No Cache-Control')
df['Content-Type'] = df['Content-Type'].fillna('Unknown Content-Type')

threshold = 13000
content_type_counts = df['Content-Type'].value_counts()
low_freq_content_types = content_type_counts[content_type_counts < threshold].index
df['Content-Type'] = df['Content-Type'].apply(lambda x: 'Other' if x in low_freq_content_types else x)
one_hot_encoder_ct = OneHotEncoder(sparse_output=False)
content_type_encoded = one_hot_encoder_ct.fit_transform(df[['Content-Type']])
content_type_encoded_df = pd.DataFrame(content_type_encoded, columns=one_hot_encoder_ct.get_feature_names_out())
print("Content-Type 列处理完成，维度：", content_type_encoded_df.shape)

# 保存 OneHotEncoder 编码器
joblib.dump(one_hot_encoder_ct, './编码器/onehot_encoder_content_type.pkl')
print("Content-Type 编码器已保存")

cache_control_features = ['no-cache', 'no-store', 'max-age', 'private', 'must-revalidate']
for feature in cache_control_features:
    df[feature] = df['Cache-Control'].apply(lambda x: 1 if feature in x else 0)
print("Cache-Control 特征处理完成，维度：", df[cache_control_features].shape)

# 处理证书相关列
print("处理证书相关列...")
df['Issuer'] = df['Issuer'].fillna('Unknown Issuer')
issuer_counts = df['Issuer'].value_counts().to_dict()
df['Issuer_Freq'] = df['Issuer'].map(issuer_counts)
print("Issuer 频率编码完成，维度：", df[['Issuer_Freq']].shape)
issuer_counts_path = './编码器/issuer_freq_dict.pkl'
joblib.dump(issuer_counts, issuer_counts_path)
print("Issuer频率字典已保存至", issuer_counts_path)


df['Not Before'] = pd.to_datetime(df['Not Before'], errors='coerce', utc=True)
df['Not After'] = pd.to_datetime(df['Not After'], errors='coerce', utc=True)
df['Validity Period'] = (df['Not After'] - df['Not Before']).dt.days
df['Validity Period'] = df['Validity Period'].fillna(df['Validity Period'].median())
print("日期相关特征处理完成，维度：", df[['Validity Period']].shape)

# 使用 OneHotEncoder 对 Signature Algorithm 进行编码
one_hot_encoder_sa = OneHotEncoder(sparse_output=False)
signature_algorithm_encoded = one_hot_encoder_sa.fit_transform(df[['Signature Algorithm']])
signature_algorithm_encoded_df = pd.DataFrame(signature_algorithm_encoded, columns=one_hot_encoder_sa.get_feature_names_out())
print("Signature Algorithm 处理完成，维度：", signature_algorithm_encoded_df.shape)

# 保存 OneHotEncoder 编码器
joblib.dump(one_hot_encoder_sa, './编码器/onehot_encoder_signature_algorithm.pkl')
print("Signature Algorithm 编码器已保存")

# 处理 Status Code 列
print("处理 Status Code 列...")
df['Status Code'] = df['Status Code'].fillna(0).astype(str)
status_code_counts = df['Status Code'].value_counts()
low_freq_status_codes = status_code_counts[status_code_counts < threshold].index
df['Status Code'] = df['Status Code'].apply(lambda x: 'Other' if x in low_freq_status_codes else x)
one_hot_encoder_sc = OneHotEncoder(sparse_output=False)
status_code_encoded = one_hot_encoder_sc.fit_transform(df[['Status Code']])
status_code_encoded_df = pd.DataFrame(status_code_encoded, columns=one_hot_encoder_sc.get_feature_names_out())
print("Status Code 列处理完成，维度：", status_code_encoded_df.shape)

# 保存 OneHotEncoder 编码器
joblib.dump(one_hot_encoder_sc, './编码器/onehot_encoder_status_code.pkl')
print("Status Code 编码器已保存")

# 处理其他特征
print("处理其他特征...")
df['Port_53_Open'] = df['Port_53_Open'].astype(int)
df['Port_853_Open'] = df['Port_853_Open'].astype(int)
df['SANs Contains DNS/DOH'] = df['SANs Contains DNS/DOH'].astype(int)
print("其他特征处理完成，维度：", df[['Port_53_Open', 'Port_853_Open', 'SANs Contains DNS/DOH']].shape)

# 合并所有处理过的特征，并保留 IP 和 DOH_Server 列
print("合并所有处理过的特征...")
all_features_df = pd.concat([
    df[['IP', 'DOH_Server']],  # 保留 IP 和 DOH_Server 列
    server_hashed_df, content_type_encoded_df, df[cache_control_features],
    status_code_encoded_df, signature_algorithm_encoded_df, df[['Port_53_Open', 'Port_853_Open', 'SANs Contains DNS/DOH', 'Validity Period', 'Issuer_Freq']]
], axis=1)
print("特征合并完成，维度：", all_features_df.shape)

# 保存所有数据
output_path = '../探测的原始数据/processed_data-1.csv'
all_features_df.to_csv(output_path, index=False)
print(f"处理过的所有数据已保存至{output_path}")

positive_samples = all_features_df[all_features_df['DOH_Server'] == 1]  # 假设 'DOH_Server' 是目标变量
positive_samples_path = '../探测的原始数据/positive_samples-1.csv'
positive_samples.to_csv(positive_samples_path, index=False)
print(f"所有正样本已保存至{positive_samples_path}")
