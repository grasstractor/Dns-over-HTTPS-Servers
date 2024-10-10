import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder
import joblib  # Used to save the encoders
import numpy as np

# Read data from CSV file
print("Reading CSV file...")
df = pd.read_csv('../raw_detection_data/final_extracted_data.csv')
print("CSV file read complete")

# Remove duplicates, excluding the IP column
print("Removing duplicates...")
df = df.drop_duplicates(subset=df.columns.difference(['IP']))
df.reset_index(drop=True, inplace=True)  # Reset index
print("Duplicate removal complete")

# Process the Server column
print("Processing the Server column...")
df['Server'] = df['Server'].fillna('Unknown Server')
hasher = FeatureHasher(input_type='string', n_features=64)
server_hashed = hasher.transform(df['Server'].apply(lambda x: [x])).toarray()
server_hashed_df = pd.DataFrame(server_hashed, columns=[f'Server_{i}' for i in range(64)])
print("Server column processing complete, dimensions: ", server_hashed_df.shape)

# Save the FeatureHasher encoder
joblib.dump(hasher, './encoders/feature_hasher_server.pkl')
print("Server encoder saved")

# Process Cache-Control and Content-Type columns
print("Processing Cache-Control and Content-Type columns...")
df['Cache-Control'] = df['Cache-Control'].fillna('No Cache-Control')
df['Content-Type'] = df['Content-Type'].fillna('Unknown Content-Type')

threshold = 13000
content_type_counts = df['Content-Type'].value_counts()
low_freq_content_types = content_type_counts[content_type_counts < threshold].index
df['Content-Type'] = df['Content-Type'].apply(lambda x: 'Other' if x in low_freq_content_types else x)
one_hot_encoder_ct = OneHotEncoder(sparse_output=False)
content_type_encoded = one_hot_encoder_ct.fit_transform(df[['Content-Type']])
content_type_encoded_df = pd.DataFrame(content_type_encoded, columns=one_hot_encoder_ct.get_feature_names_out())
print("Content-Type column processing complete, dimensions: ", content_type_encoded_df.shape)

# Save the OneHotEncoder for Content-Type
joblib.dump(one_hot_encoder_ct, './encoders/onehot_encoder_content_type.pkl')
print("Content-Type encoder saved")

cache_control_features = ['no-cache', 'no-store', 'max-age', 'private', 'must-revalidate']
for feature in cache_control_features:
    df[feature] = df['Cache-Control'].apply(lambda x: 1 if feature in x else 0)
print("Cache-Control feature processing complete, dimensions: ", df[cache_control_features].shape)

# Process certificate-related columns
print("Processing certificate-related columns...")
df['Issuer'] = df['Issuer'].fillna('Unknown Issuer')
issuer_counts = df['Issuer'].value_counts().to_dict()
df['Issuer_Freq'] = df['Issuer'].map(issuer_counts)
print("Issuer frequency encoding complete, dimensions: ", df[['Issuer_Freq']].shape)
issuer_counts_path = './encoders/issuer_freq_dict.pkl'
joblib.dump(issuer_counts, issuer_counts_path)
print("Issuer frequency dictionary saved to", issuer_counts_path)

df['Not Before'] = pd.to_datetime(df['Not Before'], errors='coerce', utc=True)
df['Not After'] = pd.to_datetime(df['Not After'], errors='coerce', utc=True)
df['Validity Period'] = (df['Not After'] - df['Not Before']).dt.days
df['Validity Period'] = df['Validity Period'].fillna(df['Validity Period'].median())
print("Date-related feature processing complete, dimensions: ", df[['Validity Period']].shape)

# Use OneHotEncoder to encode Signature Algorithm
one_hot_encoder_sa = OneHotEncoder(sparse_output=False)
signature_algorithm_encoded = one_hot_encoder_sa.fit_transform(df[['Signature Algorithm']])
signature_algorithm_encoded_df = pd.DataFrame(signature_algorithm_encoded, columns=one_hot_encoder_sa.get_feature_names_out())
print("Signature Algorithm processing complete, dimensions: ", signature_algorithm_encoded_df.shape)

# Save the OneHotEncoder for Signature Algorithm
joblib.dump(one_hot_encoder_sa, './encoders/onehot_encoder_signature_algorithm.pkl')
print("Signature Algorithm encoder saved")

# Process the Status Code column
print("Processing the Status Code column...")
df['Status Code'] = df['Status Code'].fillna(0).astype(str)
status_code_counts = df['Status Code'].value_counts()
low_freq_status_codes = status_code_counts[status_code_counts < threshold].index
df['Status Code'] = df['Status Code'].apply(lambda x: 'Other' if x in low_freq_status_codes else x)
one_hot_encoder_sc = OneHotEncoder(sparse_output=False)
status_code_encoded = one_hot_encoder_sc.fit_transform(df[['Status Code']])
status_code_encoded_df = pd.DataFrame(status_code_encoded, columns=one_hot_encoder_sc.get_feature_names_out())
print("Status Code column processing complete, dimensions: ", status_code_encoded_df.shape)

# Save the OneHotEncoder for Status Code
joblib.dump(one_hot_encoder_sc, './encoders/onehot_encoder_status_code.pkl')
print("Status Code encoder saved")

# Process other features
print("Processing other features...")
df['Port_53_Open'] = df['Port_53_Open'].astype(int)
df['Port_853_Open'] = df['Port_853_Open'].astype(int)
df['SANs Contains DNS/DOH'] = df['SANs Contains DNS/DOH'].astype(int)
print("Other feature processing complete, dimensions: ", df[['Port_53_Open', 'Port_853_Open', 'SANs Contains DNS/DOH']].shape)

# Combine all processed features, keeping the IP and DOH_Server columns
print("Combining all processed features...")
all_features_df = pd.concat([
    df[['IP', 'DOH_Server']],  # Keep IP and DOH_Server columns
    server_hashed_df, content_type_encoded_df, df[cache_control_features],
    status_code_encoded_df, signature_algorithm_encoded_df, df[['Port_53_Open', 'Port_853_Open', 'SANs Contains DNS/DOH', 'Validity Period', 'Issuer_Freq']]
], axis=1)
print("Feature combination complete, dimensions: ", all_features_df.shape)

# Save all processed data
output_path = '../raw_detection_data/processed_data-1.csv'
all_features_df.to_csv(output_path, index=False)
print(f"All processed data saved to {output_path}")

positive_samples = all_features_df[all_features_df['DOH_Server'] == 1]  # Assuming 'DOH_Server' is the target variable
positive_samples_path = '../raw_detection_data/positive_samples-1.csv'
positive_samples.to_csv(positive_samples_path, index=False)
print(f"All positive samples saved to {positive_samples_path}")
