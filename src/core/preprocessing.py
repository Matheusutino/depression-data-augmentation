import re
import pandas as pd

class DataPreprocessor:
    def __init__(self, dataset_path, questionnaire_path, dataset_name):
        self.dataset_path = dataset_path
        self.questionnaire_path = questionnaire_path
        self.dataset_name = dataset_name

    def preprocess_usernames(self, series):
        usernames = []
        for x in series:
            if isinstance(x, str):  # Check if it's a string
                # Remove all '@', strip spaces, and convert to lowercase
                username = x.strip().replace('@', '').lower()
                usernames.append(username)
            else:
                # If not a string, append an empty string (or we can choose another action)
                usernames.append('')
        return usernames

    def process_tweet(self, text):
        """Replaces mentions (@user) with @USERNAME."""
        return re.sub(r'@\w+', '@USERNAME', text)

    def load_and_preprocess_dataset(self):
        """Loads and preprocesses the train, validation, and test datasets."""
        def load_and_process(file):
            df = pd.read_csv(f"{self.dataset_path}/{file}.csv")
            df = df.dropna(subset=['caption'])  # Remove rows with NaN in 'caption' column
            df['caption'] = df['caption'].apply(self.process_tweet)
            return df

        return {
            "train": load_and_process("train"),
            "val": load_and_process("val"),
            "test": load_and_process("test")
        }

    def filter_questionnaire_data(self):
        """Filters numerical columns and the Instagram username column."""
        questionnaire = pd.read_csv(self.questionnaire_path)
        numerical_columns = [col for col in questionnaire.columns if col[0].isdigit()] + [f'Nome de usuário do {self.dataset_name.title()}']
        return questionnaire[numerical_columns]

    def preprocess_data(self, grouped_df, questionnaire_filtered):
        """Processes the grouped data and returns features and labels."""
        data = {"X": [], "y": [], "bdi_values": [], "bdi_forms": [], "usernames": []}
        
        for username, group in grouped_df.iterrows():
            captions = group['caption']
            # Apply preprocessing to 'username' and the relevant column in your dataframe
            username_processed = self.preprocess_usernames([username])[0]  # Assuming 'username' is a single value
            column_name_processed = f'Nome de usuário do {self.dataset_name.title()}'

            # Apply preprocessing to the 'Nome de usuário' column in the DataFrame
            questionnaire_filtered[column_name_processed] = self.preprocess_usernames(questionnaire_filtered[column_name_processed])

            # Filtering the dataframe based on processed username
            bdi_questionnaire = questionnaire_filtered[questionnaire_filtered[column_name_processed] == username_processed]

            # Drop the column after filtering
            bdi_questionnaire = bdi_questionnaire.drop(columns=[column_name_processed])

            #print(bdi_questionnaire)

            bdi_form = '\n'.join([
                f"{col.split('. ', 1)[1] if '. ' in col else col}: {row[col]}"
                for col in bdi_questionnaire.columns
                for _, row in bdi_questionnaire.iterrows()
            ])

            #print(bdi_form)
            
            bdi_value = group['bdi'][0]
            data["X"].append(captions)
            data["y"].append(1 if bdi_value >= 20 else 0)
            data["bdi_values"].append(bdi_value)
            data["bdi_forms"].append(bdi_form)
            data["usernames"].append(username)
        
        return data
