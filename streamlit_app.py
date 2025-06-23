# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Set the Streamlit page layout and title
st.set_page_config(page_title="Transaction Change Analyzer", layout="wide")
st.title("🔍 Transaction Change Analyzer")
st.write("Upload your transaction CSV and explore what changed between PREIMAGE and POSTIMAGE fields.")

# STEP 1: Upload the CSV file
uploaded_file = st.file_uploader("📤 Upload your transaction_data.csv", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # STEP 2: Convert PREIMAGE and POSTIMAGE strings into dictionaries
    def parse_image(image_str):
        parts = str(image_str).split('|')  # Split by |
        return {i: val for i, val in enumerate(parts)}  # Store index as key

    df['PRE_PARSED'] = df['PREIMAGE'].apply(parse_image)
    df['POST_PARSED'] = df['POSTIMAGE'].apply(parse_image)

    # STEP 3: Find the differences between PREIMAGE and POSTIMAGE
    def get_diff(pre_dict, post_dict):
        diffs = []
        for k in pre_dict:
            if pre_dict.get(k, '') != post_dict.get(k, ''):
                diffs.append((k, pre_dict.get(k, ''), post_dict.get(k, '')))
        return diffs

    df['DIFFERENCES'] = df.apply(lambda row: get_diff(row['PRE_PARSED'], row['POST_PARSED']), axis=1)

    # STEP 4: Extract DEPOT ID from the 3rd position of the string (index 2)
    def extract_depot_id(image_str):
        parts = str(image_str).split('|')
        return parts[2] if len(parts) > 2 else None

    df['DEPOT_ID'] = df['PREIMAGE'].apply(extract_depot_id)

    # STEP 5: Flatten the differences into a new DataFrame
    records = []
    for _, row in df.iterrows():
        for index, pre_val, post_val in row['DIFFERENCES']:
            records.append({
                'DEPOT_ID': row['DEPOT_ID'],
                'FIELD_INDEX': index,       # The index of the field that changed
                'PRE_VALUE': pre_val,       # Old value
                'POST_VALUE': post_val      # New value
            })

    diff_df = pd.DataFrame(records)

    if diff_df.empty:
        st.warning("⚠️ No differences found between PREIMAGE and POSTIMAGE.")
    else:
        # STEP 6: Encode DEPOT_ID, PRE_VALUE, and POST_VALUE using LabelEncoder
        label_encoders = {}
        for col in ['DEPOT_ID', 'PRE_VALUE', 'POST_VALUE']:
            le = LabelEncoder()
            diff_df[col] = le.fit_transform(diff_df[col].astype(str))
            label_encoders[col] = le

        # STEP 7: Prepare features and target for model training
        X = diff_df[['DEPOT_ID', 'PRE_VALUE']]
        y = diff_df['POST_VALUE']

        # STEP 8: Remove rare classes to avoid model errors
        value_counts = y.value_counts()
        valid_classes = value_counts[value_counts >= 2].index
        mask = y.isin(valid_classes)
        X_filtered = X[mask]
        y_filtered = y[mask]

        if len(y_filtered.unique()) < 2:
            st.error("🚫 Not enough variation to train a model. Please upload more diverse data.")
        else:
            # STEP 9: Train a Random Forest model
            X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, stratify=y_filtered, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            st.success("✅ Model trained successfully!")

            # STEP 10: User selects a DEPOT ID to analyze
            depot_ids = sorted(label_encoders['DEPOT_ID'].classes_)
            user_input_depot = st.selectbox("🔎 Select a DEPOT ID to analyze:", depot_ids)

            # STEP 11: Find all historical changes for that DEPOT ID
            encoded_depot = label_encoders['DEPOT_ID'].transform([user_input_depot])[0]
            depot_changes = diff_df[diff_df['DEPOT_ID'] == encoded_depot].copy()

            # Decode values for display
            depot_changes['DEPOT_ID'] = label_encoders['DEPOT_ID'].inverse_transform(depot_changes['DEPOT_ID'])
            depot_changes['PRE_VALUE'] = label_encoders['PRE_VALUE'].inverse_transform(depot_changes['PRE_VALUE'])
            depot_changes['POST_VALUE'] = label_encoders['POST_VALUE'].inverse_transform(depot_changes['POST_VALUE'])

            # STEP 12: Display the changes as a table
            st.subheader("📋 All Field Changes for Selected DEPOT")
            st.dataframe(depot_changes[['FIELD_INDEX', 'PRE_VALUE', 'POST_VALUE']], use_container_width=True)

            # STEP 13: Show frequency of changed fields
            st.subheader("📊 Frequency of Field Changes")
            field_change_counts = depot_changes['FIELD_INDEX'].value_counts().reset_index()
            field_change_counts.columns = ['FIELD_INDEX', 'CHANGE_COUNT']
            st.bar_chart(field_change_counts.set_index('FIELD_INDEX'))

            # STEP 14: Show top 5 changed fields and a textual summary
            top_fields = field_change_counts.head(5)
            total_changes = len(depot_changes)
            most_frequent_field = top_fields.iloc[0]['FIELD_INDEX']
            most_frequent_count = top_fields.iloc[0]['CHANGE_COUNT']
            other_fields = ", ".join(str(int(i)) for i in top_fields['FIELD_INDEX'][1:].tolist())

            st.subheader("🗣️ Summary")
            st.markdown(f"""
            - Total of **{total_changes}** changes found for **DEPOT ID `{user_input_depot}`**
            - Most frequently changed field is at **position `{int(most_frequent_field)}`** with **{int(most_frequent_count)}** changes
            - Other commonly changed fields: **{other_fields}**
            - These may indicate the most edited parts of the transaction data
            """)
