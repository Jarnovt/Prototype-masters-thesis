import graphviz
import pandas as pd
import ast

def preprocess_data_graph(df_questions, df_factual_big, df_test_results):
    df_factual_big = df_factual_big.fillna('nan')
    df_factual_big['Training'] = df_factual_big['Training'].astype(str)
    df_test_results['Removed_Methods_count'] = df_test_results['Removed_Methods'].str.len()
    df_test_results['Remaining_Methods_count'] = df_test_results['Remaining_methods'].str.len()
    #df_test_results.columns = ['Unnamed: 0', 'Attribute', 'Answer', 'Removed_Methods', 'Remaining_methods', 'Removed_Methods_count', 'Remaining_Methods_count']
    df_test_results_merged = pd.merge(df_test_results, df_questions[['Attribute', 'Possible answers']], on = "Attribute")
    return df_factual_big, df_test_results_merged

result_graph = graphviz.Digraph('Q-A', node_attr={'shape': 'plaintext'})

def process_answer_small(question_attribute, answer, df_factual):
    """
    Takes the answer to the question and updates the dataframe
    ---
    Input: Attribute, answer by user, dataframe with unmasked methods
    ---
    Output: updated dataframe
    """
    # Remove unselected rows
    mask_answer = pd.Series(False, index=df_factual.index)

    if type(answer) == str:
        if '[' in answer:
            answer = ast.literal_eval(answer)
        else:
            answer = [answer]
    for example_answer_part in answer:
        mask_answer |= df_factual[question_attribute].str.contains(example_answer_part)
    if answer == []:
        mask_answer = pd.Series(True, index=df_factual.index)
    df_factual_updated = df_factual[mask_answer]
    # Remove question
    df_factual_updated = df_factual_updated.drop(columns = [question_attribute])
    
    return df_factual_updated

def create_question_nodes(result_graph, df_results):
    """
    Takes the current graph and the decision log, adds nodes for each question and returns the graph
    """
    # First node is different from the rest:
    Root_node = df_results['Removed_Methods_count'][0] + df_results['Remaining_Methods_count'][0]
    per_filled = 1
    result_graph.node(df_results['Attribute'][0], df_results['Attribute'][0] + ": " + str(Root_node), 
                      fillcolor=f"green;{1 - per_filled}:white;{per_filled}", 
                      style="striped",
                      shape='folder')
    # Create remaining nodes
    for row in range(len(df_results) - 1):
        # Shape
        # Take the next row as node, take the remaining methods of current row as value
        per_filled = int(df_results['Remaining_Methods_count'][row])/45
        result_graph.node(str(df_results['Attribute'][row + 1]), df_results['Attribute'][row + 1] + ": " + str(df_results['Remaining_Methods_count'][row]),
                            fillcolor=f"green;{1 - per_filled}:white;{per_filled}",
                            style="striped", 
                            shape='folder')
    # Create end node
    per_filled = int(df_results['Remaining_Methods_count'][len(df_results) - 1])/45
    result_graph.node("End", "End" + ": " + str(df_results['Remaining_Methods_count'][len(df_results) - 1]),    # -1 because Python starts counting at 0
                        fillcolor=f"green;{1 - per_filled}:white;{per_filled}",
                        style="striped",
                        shape='folder')    
    return result_graph

def create_answer_nodes(result_graph, df_results, df_methods):
    """
    Takes the current graph, the decision log and the method classification.
    Adds nodes for possible answers and creates edges between questions and answers.
    Returns the graph.
    """
    # Create all nodes & edges, except for the last questions
    for row in range(len(df_results) - 1):
        # Count number of occurances possible answers
        df_values = pd.DataFrame([x for xs in list(df_methods[df_results['Attribute'][row]].str.split(',')) for x in xs]).value_counts()
        for answer in df_results['Possible answers'][row].split(','):
            # Create answer node
            if answer in df_values.index:
                per_filled = int(df_values[answer]) / 45
                result_graph.node(str(row) + answer, answer + ": " + str(int(df_values[answer])),
                            fillcolor=f"greenyellow;{1 - per_filled}:white;{per_filled}",
                            style="striped",
                            shape="tab")
            else:
                per_filled = 0
                result_graph.node(str(row) + answer, answer + ": " + "0", 
                                  fillcolor=f"greenyellow;{1 - per_filled}:white;{per_filled}",
                                  style="striped",
                                  shape = "tab")
            # If answer is chosen, connect with both previous and next question
            if answer in df_results['Answer'][row]:
                result_graph.edge(str(df_results['Attribute'][row]), str(row) + answer)         # Edge previous question -> Answer
                result_graph.edge(str(row) + answer, str(df_results['Attribute'][row + 1]))     # Edge Answer -> Next question
            # If user selected nothing, connect with both previous and next question            
            elif df_results['Answer'][row] == []:
                result_graph.edge(str(df_results['Attribute'][row]), str(row) + answer)         # Edge previous question -> Answer
                result_graph.edge(str(row) + answer, str(df_results['Attribute'][row + 1]))     # Edge Answer -> Next question
            # If answer is not chosen, connect with only previous question
            else:
                result_graph.edge(str(df_results['Attribute'][row]), str(row) + answer, style = 'dashed')   # Edge previous question -> Answer
                result_graph.node(str(row) + answer,
                            fillcolor=f"#eeffd5;{1 - per_filled}:white;{per_filled}",
                            style="striped",    
                            shape="tab")
        df_methods = process_answer_small(df_results['Attribute'][row], df_results['Answer'][row], df_methods)

    # Create nodes and edges for the last row in a similar fashion
    last_row = len(df_results)-1
    df_values = pd.DataFrame([x for xs in list(df_methods[df_results['Attribute'][last_row]].str.split(',')) for x in xs]).value_counts()
    for answer in df_results['Possible answers'][last_row].split(','):

        if answer in df_values.index:
            per_filled = int(df_values[answer]) / 45
            result_graph.node(str(last_row) + answer, answer + ": " + str(int(df_values[answer])),
                            fillcolor=f"greenyellow;{1 - per_filled}:white;{per_filled}",
                            style="striped",
                            shape="tab")
        else:
            per_filled = 0
            result_graph.node(str(last_row) + answer, answer + ": " + "0",
                              fillcolor=f"greenyellow;{1 - per_filled}:white;{per_filled}",
                              style="striped",
                              shape="tab")

        if answer in df_results['Answer'][last_row]:
            result_graph.edge(str(df_results['Attribute'][last_row]), str(last_row) + answer)
            result_graph.edge(str(last_row) + answer, "End") 
        elif df_results['Answer'][last_row] == []:
            result_graph.edge(str(df_results['Attribute'][last_row]), str(last_row) + answer)
            result_graph.edge(str(last_row) + answer, "End") 
        else:
            result_graph.edge(str(df_results['Attribute'][last_row]), str(last_row) + answer, style = 'dashed')
            result_graph.node(str(last_row) + answer,
                                fillcolor=f"#eeffd5;{1 - per_filled}:white;{per_filled}",
                                style="striped",
                                shape="tab")

    return result_graph

def create_single_question(asked_question, df_factual, df_questions):
    # Graph initialization
    small_graph = graphviz.Digraph('Q-A', node_attr={'shape': 'plaintext'})
    # Shape
    small_graph.attr('node', shape='box')
    # Question node
    small_graph.node(asked_question[0], asked_question[0] + ": " + str(len(df_factual)))
    # Count number of occurances possible answers
    df_values_small = pd.DataFrame([x for xs in list(df_factual[asked_question[0]].astype(str).str.split(',')) for x in xs]).value_counts()
    # Answer nodes
    answer_options_graph = df_questions.loc[df_questions["Attribute"] == asked_question[0], "Possible answers"].iloc[0].split(',')
    for a_option in answer_options_graph:
        if a_option in df_values_small.index:
            small_graph.node(a_option, a_option + ": " + str(int(df_values_small[a_option])))
        else:
            small_graph.node(a_option, a_option + ": " + "0")     
        # Connections
        small_graph.edge(asked_question[0], a_option)         # Edge previous question -> Answer
    return small_graph

def create_old_dataframe(decision_log, df_original, selected_question):
    """
    Takes a selected question and creates the dataframe at that point in the decision process
    ---
    Input: Decision log generated, original dataframe, selected question
    ---
    Output: Backtracked dataframe and decision log
    """
    decision_log = decision_log.reset_index(drop=True)
    # Find decision the index of the selected question in the decision log
    index_selected_question = decision_log[decision_log['Attribute'] == selected_question].index.values
    # Select up to but not including answer: this question is selected to go back to
    df_test_results_selected = decision_log.iloc[0:index_selected_question[0]]
    removed_selected = df_test_results_selected['Removed_Methods'].values
    # Create a flat list of all the removed methods
    removed_selected_flat = list()
    for list_removed_method in removed_selected:
        # Skips if NaN
        if type(list_removed_method) == list:
            #list_removed_method = list_removed_method.split(',')
            for removed_method in list_removed_method:
                removed_selected_flat.append(removed_method)

    # Drops all removed methods from the originial dataframe
    df_backtracked = df_original.copy()
    for row_number in range(len(df_original)):
        if df_backtracked['Name'][row_number] in removed_selected_flat:
            df_backtracked = df_backtracked.drop([row_number])
    # Create decision log
    decision_log_backtracked = decision_log.iloc[0:index_selected_question[0]]
    # Drop columns already answered
    df_backtracked = df_backtracked.drop(columns = decision_log_backtracked["Attribute"])
    return df_backtracked, decision_log_backtracked

def align_graph(result_graph, df_results):
    """
    Should align the graph
    """
    for i, attr in enumerate(df_results['Attribute']):
        with result_graph.subgraph() as s:
            s.attr(rank='same')
            s.node(str(attr))

    # End node aligned as last question
    with result_graph.subgraph() as s:
        s.attr(rank='same')
        s.node("End")
    for i in range(len(df_results) - 1):
        result_graph.edge(
            str(df_results['Attribute'][i]),
            str(df_results['Attribute'][i + 1]),
            style='invis',
            weight='100'
        )

    result_graph.edge(
        str(df_results['Attribute'].iloc[-1]),
        "End",
        style='invis',
        weight='10'
    )
    return result_graph