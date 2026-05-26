import pandas as pd
import numpy as np
import math

def load_method_data():
    df_factual = pd.read_excel("Data/Methods_final - Working Copy.xlsx")
    # for row in df_factual.iterrows():
    #     row[1][0] = str(row[1][0]).split('","')
    # df_factual = pd.DataFrame(list(df_factual[0][1:]), columns = list(df_factual[0])[0])
    df_factual = df_factual.fillna('nan')
    df_factual = df_factual.drop(columns = ['Unnamed: 0'])
    return df_factual

def load_question_data():
    df_questions = pd.read_excel("Data/Methods_final - Working Copy.xlsx", sheet_name = "Questions")
    return df_questions

def alternate_IG(Df:pd.DataFrame, column_interest:str) -> float:
    """
    Calculates a naive version of information gain
    ---
    Input: Dataframe with methods, column name of interest
    ---
    Output: information gain for the column of interest
    """
    # Define column information
    total_entries = sum(list(Df[column_interest].value_counts()))       # Counts the # of entries in the column
    column_set = set(Df[column_interest])                               # Creates a set with all unique entries
    values_dict = dict(Df[column_interest].value_counts())              # Creates a dictionary with # occurances per unique entire
    entropy_dict = dict()

    # Calculates fraction of methods dropped each time
    for attribute in column_set:
        entropy_dict[attribute] = values_dict[attribute]/total_entries
    average_ig = sum(entropy_dict.values())/len(entropy_dict)           # Calculates average fraction of attributes
    relevance_indicator = abs(average_ig - 0.5)                         # Centers it around the desired value (0.5)
    return relevance_indicator

def calculate_IG(Df:pd.DataFrame, column_interest:str) -> float:
    """
    Calculates the information gain for a specified column
    ---
    Input: Dataframe with methods, column name of interest
    ---
    Output: information gain for the column of interest
    """
    # Define column information
    total_entries = sum(list(Df[column_interest].value_counts()))       # Counts the # of entries in the column
    column_set = set(Df[column_interest])                               # Creates a set with all unique entries
    values_dict = dict(Df[column_interest].value_counts())              # Creates a dictionary with # occurances per unique entire

    # Define base
    base = 2
    entropy_dict = dict()

    # Compute entropy of the parent node
    parent_entropy = math.log(total_entries, base) 

    # Compute entropy for each child
    for attribute in column_set:
        entropy_dict[attribute] = -1 * (values_dict[attribute]/total_entries)*math.log(values_dict[attribute]/total_entries, base)     # Stores entropy for each child in a dictionary

    # Compute information gain
    IG_column = parent_entropy - sum(entropy_dict.values())

    return IG_column

def find_decision_classes(Df:pd.DataFrame, excluded_columns = list):
    """
    Finds indistuingishable rows and groups them into one decision class
    ---
    Input: Dataframe with methods, list with excluded columns
    ---
    Output: Returns a list with all unique decision classes
    """
    new_decision_classes = []               # List of new decision classes
    decision_class_number = 0               # Keeps track of the number of decision classes
    decision_classes = []                   # Links decision classes with methods
    mask_for_df = []
    
    # Drop attributes with no questions attached
    if 'Decision_class' in Df.columns:
        excluded_columns.append('Decision_class')
    Df_check = Df.drop(columns = excluded_columns)
    
    # For each current decision class, check if it is already in the list, if not, add it
    for current_class in range(len(Df)):
        current_class_list = list(Df_check.iloc[current_class])                 # Turns decision class into a list
        if current_class_list not in new_decision_classes:                      # Compare to current class
            new_decision_classes.append(list(Df_check.iloc[current_class]))     # Add if not there yet
            decision_classes.append((decision_class_number, Df['Name'].iloc[current_class]))    # Adds decisionclass-value pair
            decision_class_number += 1
            mask_for_df.append(True)
        else:
            decision_classes.append((new_decision_classes.index(current_class_list), Df['Name'].iloc[current_class]))   # Adds decisionclass-value pair
            mask_for_df.append(False)
    return new_decision_classes, decision_classes, mask_for_df

def find_sc(Df:pd.DataFrame, Df_Q:pd.DataFrame, excluded_columns:list):
    """
    Finds the column with the most information gain
    ---
    Input: Dataframe with methods, dataframe with questions, list with excluded columns
    ---
    Output: Returns the question with the most information gain 
    """
    highest_IG = ('Temp', 1000000)

    # TODO this should be redundant in the future
    # Remove columns which are in the question list but not in the DataFrame
    excluded_attributes = list(set(Df_Q['Attribute'].to_list()) - set(Df.columns.to_list()))
    for excluded_a in excluded_attributes:
        Df_Q = Df_Q[Df_Q['Attribute'] != excluded_a]

    # Adds restriction to which questions can be asked
    # Check if in the remaining questions there still is a question of a higher rank, otherwise move on
    if len(Df_Q[Df_Q['Importance'] == 'Need']) > 0:
        att_int_list = list(Df_Q[Df_Q['Importance'] == 'Need']['Attribute'])
    elif len(Df_Q[Df_Q['Importance'] == 'Need']) == 0 and len(Df_Q[Df_Q['Importance'] == 'Want']) > 0:
        att_int_list = list(Df_Q[Df_Q['Importance'] == 'Want']['Attribute'])
    else:
        att_int_list = list(Df_Q[Df_Q['Importance'] == 'Nice to have']['Attribute'])

    for att_int in att_int_list:  # Exclude previously defined unsuitable columns
        att_int_IG = alternate_IG(Df, att_int)
        if att_int_IG < highest_IG[1]:                  # Check if the new information gain is higher
            highest_IG = (att_int, att_int_IG)
    return highest_IG[0]

def find_question(df_factual, df_questions, excluded_columns):
    """
    Calculates the next question by finding decision classes and then applying the mask to the dataframe and 
    finds the question with the highest information gain accordingly
    ---
    Input: Dataframe with methods, dataframe with questions, list with excluded columns
    ---
    Output: Attribute, Corresponding question, possible answers and amount of answers
    """

    new_decision_classes, decision_classes, mask_for_df = find_decision_classes(df_factual, excluded_columns)
    # Apply a mask to get only 1 method per decision class
    question_attribute = find_sc(df_factual[mask_for_df], df_questions, excluded_columns)
    # Find question, answer, amount of answers and get the string
    question = list(df_questions[df_questions['Attribute'] == question_attribute]['Question(s)'])[0]
    possible_answers = list(df_questions[df_questions['Attribute'] == question_attribute]['Possible answers'])[0]
    possible_amount_answers = list(df_questions[df_questions['Attribute'] == question_attribute]['How many answers'])[0]
    translation_answers = list(df_questions[df_questions['Attribute'] == question_attribute]['Translation'])[0]
    return (question_attribute, question, possible_answers, possible_amount_answers, translation_answers)

def process_answer(question_attribute, answer, df_factual, decision_log, current_step):
    """
    Takes the answer to the question and updates the decision_log, and dataframe
    ---
    Input: Attribute, answer by user, dataframe with unmasked methods, decision log
    ---
    Output: updated decision log method dataframe
    """
    # If exclude -> exclude selected answer
    # If any -> not any also contains any
    # If full -> full answer needs to be contained
    # If partial -> use as before
    # if 1 -> needs to be contained
    # if not 1 -> stack filters
    # Remove unselected rows
    mask_answer = pd.Series(False, index=df_factual.index)
    if type(answer) == str:
        answer = [answer]
    for example_answer_part in answer:
        print(example_answer_part)
        mask_answer |= df_factual[question_attribute].str.contains(example_answer_part)
    if answer == []:
        mask_answer = pd.Series(True, index=df_factual.index)
    df_factual_updated = df_factual[mask_answer]
    # Update decision log
    df_factual_removed = df_factual[~mask_answer]
    decision_log.loc[current_step] = [question_attribute, answer, df_factual_removed['Name'].to_list(), df_factual_updated['Name'].to_list()]
    # Remove question
    df_factual_updated = df_factual_updated.drop(columns = [question_attribute])
    
    return df_factual_updated, decision_log

def process_answer_updated(question_attribute, answer, df_factual, df_questions, decision_log, current_step):
    """
    Takes the answer to the question and updates the decision_log, and dataframe
    ---
    Input: Attribute, answer by user, dataframe with unmasked methods, decision log, dataframe with questions
    ---
    Output: updated decision log method dataframe
    """
    df_factual_process = df_factual.copy()
    df_factual_process = df_factual_process.reset_index(drop = True)
    att_oi = df_questions[df_questions['Attribute'] == question_attribute]
    if type(answer) == str:
        answer = [answer]

    #If exclude -> exclude selected answer
    if att_oi['Type'].values[0] == 'Exclude':
        mask_answer = pd.Series(False, index=df_factual_process.index)
        # Find all rows that match part of the answer
        for example_answer_part in answer:
            mask_answer |= df_factual_process[question_attribute].str.contains(example_answer_part)
        # Exclude these rows
        mask_answer = ~mask_answer
        if answer == []:
            mask_answer = pd.Series(True, index=df_factual_process.index)

    # If partial -> include answer
    elif att_oi['Type'].values[0] == 'Partial':
        mask_answer = pd.Series(False, index=df_factual_process.index)
        # Find all rows that match part of the answer
        for example_answer_part in answer:
            mask_answer |= df_factual_process[question_attribute].str.contains(example_answer_part)
        if answer == []:
            mask_answer = pd.Series(True, index=df_factual_process.index)

    # If any -> not any also includes any
    elif att_oi['Type'].values[0] == 'Any':
        mask_answer = pd.Series(False, index=df_factual_process.index)
        if 'Any' not in answer:
            answer.append('Any')
        for example_answer_part in answer:
            mask_answer |= df_factual_process[question_attribute].str.contains(example_answer_part)

    # If full -> full answer needs to be contained
    elif att_oi['Type'].values[0] == 'Full':
        mask_answer = pd.Series(False, index=df_factual_process.index)
        if answer != []:
            for col_index in range(len(df_factual_process)):
                if False not in [q1 in answer for q1 in df_factual_process[question_attribute][col_index].split(',')]:  # Errors if a method does not contain a value
                    mask_answer[col_index] = True

    df_factual_updated = df_factual_process[mask_answer]
    # Update decision log
    df_factual_removed = df_factual_process[~mask_answer]
    decision_log.loc[current_step] = [question_attribute, answer, df_factual_removed['Name'].to_list(), df_factual_updated['Name'].to_list()]
    # Remove question
    df_factual_updated = df_factual_updated.drop(columns = [question_attribute])
    
    return df_factual_updated, decision_log