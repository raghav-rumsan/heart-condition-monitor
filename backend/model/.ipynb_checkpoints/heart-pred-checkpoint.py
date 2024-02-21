{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Import necessary libraries\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "import joblib\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import spacy\n",
    "from spacy.lang.en.stop_words import STOP_WORDS\n",
    "\n",
    "def spacy_summarizer(text, num_tokens=3):\n",
    "    # Load the spaCy English model\n",
    "    nlp = spacy.load(\"en_core_web_sm\")\n",
    "\n",
    "    # Process the text using spaCy\n",
    "    doc = nlp(text)\n",
    "\n",
    "    # Extract noun phrases and remove stopwords\n",
    "    noun_phrases = [\n",
    "        chunk.text.replace(\" \", \"_\") \n",
    "        for chunk in doc.noun_chunks \n",
    "        if chunk.text.lower() not in STOP_WORDS\n",
    "    ]\n",
    "\n",
    "    # If there are noun phrases, use them as summary\n",
    "    if noun_phrases:\n",
    "        summary = \"_\".join(noun_phrases[:num_tokens])\n",
    "    else:\n",
    "        # If no noun phrases, use the first few tokens\n",
    "        summary = \"_\".join([token.text for token in doc][:num_tokens])\n",
    "\n",
    "    return summary.lower()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                         Timestamp Your name  Your gender  Your age   \\\n",
      "0  2019/07/03 11:48:07 PM GMT+5:30    Parkavi       Female  19 to 25   \n",
      "1  2019/07/03 11:51:22 PM GMT+5:30   Nithilaa       Female  19 to 25   \n",
      "\n",
      "   How important is exercise to you ?  \\\n",
      "0                                   2   \n",
      "1                                   4   \n",
      "\n",
      "  How do you describe your current level of fitness ?  \\\n",
      "0                                               Good    \n",
      "1                                          Very good    \n",
      "\n",
      "  How often do you exercise?  \\\n",
      "0                      Never   \n",
      "1                      Never   \n",
      "\n",
      "  What barriers, if any, prevent you from exercising more regularly?           (Please select all that apply)  \\\n",
      "0    I don't have enough time;I can't stay motivated                                                            \n",
      "1     I don't have enough time;I'll become too tired                                                            \n",
      "\n",
      "  What form(s) of exercise do you currently participate in ?                        (Please select all that apply)  \\\n",
      "0                            I don't really exercise                                                                 \n",
      "1                        Walking or jogging;Swimming                                                                 \n",
      "\n",
      "  Do you exercise ___________ ?  \\\n",
      "0       I don't really exercise   \n",
      "1                  With a group   \n",
      "\n",
      "  What time if the day do you prefer to exercise?  \\\n",
      "0                                   Early morning   \n",
      "1                                   Early morning   \n",
      "\n",
      "  How long do you spend exercising per day ?  \\\n",
      "0                    I don't really exercise   \n",
      "1                    I don't really exercise   \n",
      "\n",
      "  Would you say you eat a healthy balanced diet ?  \\\n",
      "0                                      Not always   \n",
      "1                                      Not always   \n",
      "\n",
      "  What prevents you from eating a healthy balanced diet, If any?                         (Please select all that apply)  \\\n",
      "0  Ease of access to fast food;Temptation and cra...                                                                      \n",
      "1  Ease of access to fast food;Temptation and cra...                                                                      \n",
      "\n",
      "   How healthy do you consider yourself?  \\\n",
      "0                                      3   \n",
      "1                                      4   \n",
      "\n",
      "  Have you ever recommended your friends to follow a fitness routine?  \\\n",
      "0                                                Yes                    \n",
      "1                                                Yes                    \n",
      "\n",
      "  Have you ever purchased a fitness equipment?  \\\n",
      "0                                           No   \n",
      "1                                           No   \n",
      "\n",
      "  What motivates you to exercise?         (Please select all that applies )  \n",
      "0  I'm sorry ... I'm not really interested in exe...                         \n",
      "1  I want to be fit;I want to be flexible;I want ...                         \n"
     ]
    }
   ],
   "source": [
    "# Load the dataset\n",
    "df = pd.read_csv('../datasets/fitness_survey.csv')\n",
    "print(df.head(2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "def generate_column_mapping(columns):\n",
    "    \"\"\"\n",
    "    Generate a mapping of old column names to summarized new column names.\n",
    "\n",
    "    Parameters:\n",
    "    - columns (list): List of old column names.\n",
    "\n",
    "    Returns:\n",
    "    - dict: Mapping of old column names to summarized new column names.\n",
    "    \"\"\"\n",
    "    column_mapping = {column: spacy_summarizer(column) for column in columns}\n",
    "    return column_mapping\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'Timestamp': 'timestamp', 'Your name ': 'your_name', 'Your gender ': 'your_gender', 'Your age ': 'your_age', 'How important is exercise to you ?': 'exercise', 'How do you describe your current level of fitness ?': 'your_current_level_fitness', 'How often do you exercise?': 'how_often_do', 'What barriers, if any, prevent you from exercising more regularly?           (Please select all that apply)': 'what_barriers', 'What form(s) of exercise do you currently participate in ?                        (Please select all that apply)': 'what_form(s_exercise', 'Do you exercise ___________ ?': 'do_you_exercise', 'What time if the day do you prefer to exercise?': 'what_time_if', 'How long do you spend exercising per day ?': 'day', 'Would you say you eat a healthy balanced diet ?': 'a_healthy_balanced_diet', 'What prevents you from eating a healthy balanced diet, If any?                         (Please select all that apply)': 'a_healthy_balanced_diet', 'How healthy do you consider yourself?': 'how_healthy_do', 'Have you ever recommended your friends to follow a fitness routine?': 'your_friends_a_fitness_routine', 'Have you ever purchased a fitness equipment?': 'a_fitness_equipment', 'What motivates you to exercise?         (Please select all that applies )': 'what_motivates_you'}\n",
      "                         timestamp your_name your_gender  your_age  exercise  \\\n",
      "0  2019/07/03 11:48:07 PM GMT+5:30   Parkavi      Female  19 to 25         2   \n",
      "1  2019/07/03 11:51:22 PM GMT+5:30  Nithilaa      Female  19 to 25         4   \n",
      "\n",
      "  your_current_level_fitness how_often_do  \\\n",
      "0                       Good        Never   \n",
      "1                  Very good        Never   \n",
      "\n",
      "                                     what_barriers  \\\n",
      "0  I don't have enough time;I can't stay motivated   \n",
      "1   I don't have enough time;I'll become too tired   \n",
      "\n",
      "          what_form(s_exercise          do_you_exercise   what_time_if  \\\n",
      "0      I don't really exercise  I don't really exercise  Early morning   \n",
      "1  Walking or jogging;Swimming             With a group  Early morning   \n",
      "\n",
      "                       day a_healthy_balanced_diet  \\\n",
      "0  I don't really exercise              Not always   \n",
      "1  I don't really exercise              Not always   \n",
      "\n",
      "                             a_healthy_balanced_diet  how_healthy_do  \\\n",
      "0  Ease of access to fast food;Temptation and cra...               3   \n",
      "1  Ease of access to fast food;Temptation and cra...               4   \n",
      "\n",
      "  your_friends_a_fitness_routine a_fitness_equipment  \\\n",
      "0                            Yes                  No   \n",
      "1                            Yes                  No   \n",
      "\n",
      "                                  what_motivates_you  \n",
      "0  I'm sorry ... I'm not really interested in exe...  \n",
      "1  I want to be fit;I want to be flexible;I want ...  \n"
     ]
    }
   ],
   "source": [
    "# Extract the column names\n",
    "df_columns = df.columns.tolist()\n",
    "\n",
    "# Generate column mapping with summarized names\n",
    "column_mapping_result = generate_column_mapping(df_columns)\n",
    "\n",
    "print(column_mapping_result)\n",
    "\n",
    "# # Replace the DataFrame columns with the summarized versions\n",
    "df.rename(columns=column_mapping_result, inplace=True)\n",
    "\n",
    "# # Print the DataFrame with updated column names\n",
    "print(df.head(2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                         timestamp your_name your_gender  your_age  exercise  \\\n",
      "0  2019/07/03 11:48:07 PM GMT+5:30   Parkavi      Female  19 to 25         2   \n",
      "1  2019/07/03 11:51:22 PM GMT+5:30  Nithilaa      Female  19 to 25         4   \n",
      "\n",
      "  your_current_level_fitness how_often_do  \\\n",
      "0                       Good        Never   \n",
      "1                  Very good        Never   \n",
      "\n",
      "                                     what_barriers  \\\n",
      "0  I don't have enough time;I can't stay motivated   \n",
      "1   I don't have enough time;I'll become too tired   \n",
      "\n",
      "          what_form(s_exercise          do_you_exercise   what_time_if  \\\n",
      "0      I don't really exercise  I don't really exercise  Early morning   \n",
      "1  Walking or jogging;Swimming             With a group  Early morning   \n",
      "\n",
      "                       day a_healthy_balanced_diet  \\\n",
      "0  I don't really exercise              Not always   \n",
      "1  I don't really exercise              Not always   \n",
      "\n",
      "                             a_healthy_balanced_diet  how_healthy_do  \\\n",
      "0  Ease of access to fast food;Temptation and cra...               3   \n",
      "1  Ease of access to fast food;Temptation and cra...               4   \n",
      "\n",
      "  your_friends_a_fitness_routine a_fitness_equipment  \\\n",
      "0                            Yes                  No   \n",
      "1                            Yes                  No   \n",
      "\n",
      "                                  what_motivates_you  \n",
      "0  I'm sorry ... I'm not really interested in exe...  \n",
      "1  I want to be fit;I want to be flexible;I want ...  \n"
     ]
    }
   ],
   "source": [
    "# List of columns to remove\n",
    "columns_to_remove = list(set(df.columns) - set(column_mapping_result.values()))\n",
    "\n",
    "# Remove unnecessary columns from the DataFrame\n",
    "df = df.drop(columns=columns_to_remove)\n",
    "\n",
    "# Display the updated DataFrame\n",
    "print(df.head(2))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "y should be a 1d array, got an array of shape (545, 2) instead.",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[114], line 9\u001b[0m\n\u001b[1;32m      6\u001b[0m categorical_columns \u001b[38;5;241m=\u001b[39m [\u001b[38;5;124m'\u001b[39m\u001b[38;5;124myour_gender\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124myour_current_level_fitness\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mhow_often_do\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mwhat_time_if\u001b[39m\u001b[38;5;124m'\u001b[39m,\n\u001b[1;32m      7\u001b[0m                         \u001b[38;5;124m'\u001b[39m\u001b[38;5;124ma_healthy_balanced_diet\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124myour_friends_a_fitness_routine\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124ma_fitness_equipment\u001b[39m\u001b[38;5;124m'\u001b[39m]\n\u001b[1;32m      8\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m column \u001b[38;5;129;01min\u001b[39;00m categorical_columns:\n\u001b[0;32m----> 9\u001b[0m     df[column] \u001b[38;5;241m=\u001b[39m label_encoder\u001b[38;5;241m.\u001b[39mfit_transform(df[column])\n\u001b[1;32m     11\u001b[0m \u001b[38;5;66;03m# Convert timestamp to datetime\u001b[39;00m\n\u001b[1;32m     12\u001b[0m df[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mtimestamp\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mto_datetime(df[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mtimestamp\u001b[39m\u001b[38;5;124m'\u001b[39m])\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/sklearn/utils/_set_output.py:140\u001b[0m, in \u001b[0;36m_wrap_method_output.<locals>.wrapped\u001b[0;34m(self, X, *args, **kwargs)\u001b[0m\n\u001b[1;32m    138\u001b[0m \u001b[38;5;129m@wraps\u001b[39m(f)\n\u001b[1;32m    139\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mwrapped\u001b[39m(\u001b[38;5;28mself\u001b[39m, X, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[0;32m--> 140\u001b[0m     data_to_wrap \u001b[38;5;241m=\u001b[39m f(\u001b[38;5;28mself\u001b[39m, X, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[1;32m    141\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(data_to_wrap, \u001b[38;5;28mtuple\u001b[39m):\n\u001b[1;32m    142\u001b[0m         \u001b[38;5;66;03m# only wrap the first output for cross decomposition\u001b[39;00m\n\u001b[1;32m    143\u001b[0m         return_tuple \u001b[38;5;241m=\u001b[39m (\n\u001b[1;32m    144\u001b[0m             _wrap_data_with_container(method, data_to_wrap[\u001b[38;5;241m0\u001b[39m], X, \u001b[38;5;28mself\u001b[39m),\n\u001b[1;32m    145\u001b[0m             \u001b[38;5;241m*\u001b[39mdata_to_wrap[\u001b[38;5;241m1\u001b[39m:],\n\u001b[1;32m    146\u001b[0m         )\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/sklearn/preprocessing/_label.py:114\u001b[0m, in \u001b[0;36mLabelEncoder.fit_transform\u001b[0;34m(self, y)\u001b[0m\n\u001b[1;32m    101\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mfit_transform\u001b[39m(\u001b[38;5;28mself\u001b[39m, y):\n\u001b[1;32m    102\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Fit label encoder and return encoded labels.\u001b[39;00m\n\u001b[1;32m    103\u001b[0m \n\u001b[1;32m    104\u001b[0m \u001b[38;5;124;03m    Parameters\u001b[39;00m\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    112\u001b[0m \u001b[38;5;124;03m        Encoded labels.\u001b[39;00m\n\u001b[1;32m    113\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[0;32m--> 114\u001b[0m     y \u001b[38;5;241m=\u001b[39m column_or_1d(y, warn\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n\u001b[1;32m    115\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mclasses_, y \u001b[38;5;241m=\u001b[39m _unique(y, return_inverse\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n\u001b[1;32m    116\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m y\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/sklearn/utils/validation.py:1245\u001b[0m, in \u001b[0;36mcolumn_or_1d\u001b[0;34m(y, dtype, warn)\u001b[0m\n\u001b[1;32m   1234\u001b[0m         warnings\u001b[38;5;241m.\u001b[39mwarn(\n\u001b[1;32m   1235\u001b[0m             (\n\u001b[1;32m   1236\u001b[0m                 \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mA column-vector y was passed when a 1d array was\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m   1241\u001b[0m             stacklevel\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m2\u001b[39m,\n\u001b[1;32m   1242\u001b[0m         )\n\u001b[1;32m   1243\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m _asarray_with_order(xp\u001b[38;5;241m.\u001b[39mreshape(y, (\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m,)), order\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mC\u001b[39m\u001b[38;5;124m\"\u001b[39m, xp\u001b[38;5;241m=\u001b[39mxp)\n\u001b[0;32m-> 1245\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[1;32m   1246\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124my should be a 1d array, got an array of shape \u001b[39m\u001b[38;5;132;01m{}\u001b[39;00m\u001b[38;5;124m instead.\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;241m.\u001b[39mformat(shape)\n\u001b[1;32m   1247\u001b[0m )\n",
      "\u001b[0;31mValueError\u001b[0m: y should be a 1d array, got an array of shape (545, 2) instead."
     ]
    }
   ],
   "source": [
    "# Handle missing values (if any)\n",
    "df = df.dropna()\n",
    "\n",
    "# Convert categorical columns to numerical using Label Encoding\n",
    "label_encoder = LabelEncoder()\n",
    "categorical_columns = ['your_gender', 'your_current_level_fitness', 'how_often_do', 'what_time_if',\n",
    "                        'a_healthy_balanced_diet', 'your_friends_a_fitness_routine', 'a_fitness_equipment']\n",
    "for column in categorical_columns:\n",
    "    df[column] = label_encoder.fit_transform(df[column])\n",
    "\n",
    "# Convert timestamp to datetime\n",
    "df['timestamp'] = pd.to_datetime(df['timestamp'])\n",
    "\n",
    "# Extract features and target variable\n",
    "X = df.drop(['what_motivates_you'], axis=1)  # Features\n",
    "y = df['what_motivates_you']  # Target variable\n",
    "\n",
    "# Split the data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Standardize the features (optional, depends on the model you choose)\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
