import pandas as pd


def load_approaches_data():
    df_approaches = pd.read_excel("Data/evaluation_metrics.xlsx", sheet_name = "Methods (2)")
    return df_approaches

def find_no_req(df_approaches):
    """
    Find all the approaches without requirements
    """
    potential_approaches = df_approaches[df_approaches['Requirements'] == 'noreq']
    return potential_approaches

def remove_misreq(df_approaches, df_methods, selected_method):
    """
    Removes all approaches that do not fit the attributes of the selected method
    ---
    Input: originial dataframe with all approaches, dataframe with information on all methods, selected method
    ---
    Output: dataframe with potentially applicable approaches
    """

    df_selected_method = df_methods[df_methods["Name"] == selected_method]
    df_approaches_req = df_approaches[~df_approaches['Requirements'].str.contains('noreq')]
    df_approaches_req = df_approaches_req.reset_index()
    df_approaches_req['Requirements_split'] = df_approaches_req['Requirements'].str.split(',')
    for row_count in range(len(df_approaches_req)):
        for req in df_approaches_req['Requirements_split'].loc[row_count]:
            if "Requirement" in req:
                continue
            else:
                req_name = req.split(':')[0]
                req_value = req.split(':')[1]
                if req_value in df_selected_method[req_name].values[0].split(','):
                    continue
                else:
                    df_approaches_req = df_approaches_req.drop(row_count)
                    break

    return df_approaches_req

def find_additional_questions(df_approaches_modified):
    """
    Finds all additional requirements
    ---
    Input: dataframe with all applicable approaches
    ---
    Output: additional requirements to ask
    """

    # Find all approaches with requirements not in the decision tree
    specific_approaches = df_approaches_modified[df_approaches_modified['Requirements'].str.contains('Requirement')]
    specific_approaches['Requirements_split'] = specific_approaches['Requirements'].str.split(',')
    requirement_list = specific_approaches['Requirements_split'].to_list()
    requirement_list_flat = list()
    for l1 in requirement_list:
        for element in l1:
            requirement_list_flat.append(element)
    requirement_only_list = [x for x in requirement_list_flat if 'Requirement' in x]
    requirement_only_list = list(set([req1.replace('Requirement:', '') for req1 in requirement_only_list]))
    return requirement_only_list

def remove_add_misreq(df_approaches_modified, selected_questions):
    """
    Removes all approaches that do not fit the attributes of the selected additional requirements
    ---
    Input: dataframe with all applicable approaches, selected requirements
    ---
    Output: dataframe with applicable approaches
    """
    mask_answer = pd.Series(False, index=df_approaches_modified.index)
    # Keep all the approaches which contain the selected requirement
    # NOTE: Doesn't work if there are 2 additional requirements for a method and 1 is not met. Currently, there are none with 2 additional requirements.
    for question in selected_questions:
        mask_answer += df_approaches_modified["Requirements"].str.contains(question)
    # Keep all the approaches without additional requirements as well
    mask_answer += ~df_approaches_modified["Requirements"].str.contains("Requirement")
    # Create the final dataframe
    df_approaches_final = df_approaches_modified[mask_answer]
    return df_approaches_final