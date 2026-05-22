import streamlit as st
import pandas as pd
import numpy as np
import math

from dashboard_functions import *
from backtracking_functions import *
from metric_functions import *

st.set_page_config(layout="wide")
st.markdown("""
    <style>
        /* Background color */
        .stApp {
            background-color: #f5f5f5;  /* light background */
            color: #222222;             /* dark text */
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #222222 !important;
        }

        /* Widgets text */
        .stMarkdown, .stTextInput, .stSelectbox, label, .stCheckbox {
            color: #ffffff !important;
        }

        /* BUTTON TEXT COLOR */
        .stButton > button {
        color: #ffffff !important;   /* light text */
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)


if "app_started" not in st.session_state:
    st.session_state.app_started = False

# Only runs once
if not st.session_state.app_started:
    st.title("Welcome to the dynamic decision tree :deciduous_tree:")
    
    st.markdown(
        """
        <p style="color:#000000;">
        <b>How to use</b><br>
        - Click the answer(s) most fitting your problem<br>
        - Some questions allow for multiple answers, in that case select all that apply<br>
        - Some questions allow no answer to be given. If you don't know the answer, you can skip it by leaving it empty<br>
        - You can backtrack through the answers by opening the decision tree and selecting the question you want to go back to
        </p>
        """,
        unsafe_allow_html=True
    )


    if st.button("Start the dynamic decision tree"):
        st.session_state.app_started = True
        st.rerun()

    st.markdown(
        """
    <p style="color:#71797E;">
    Created by Jarno van Tilt as part of my masters thesis <br>
    Thesis link: https://research.tue.nl/en/studentTheses/ <br>
    Code: https://github.com/? <br>
    </p>
    """,
    unsafe_allow_html=True)


else:
    col1, col2 = st.columns([7, 3])  # 70% / 30%
    # Initialize everything
    if "current_index" not in st.session_state:
        st.session_state.excluded_columns = {'Name', 'Class', 'Zetoro', 'Tunable Paramaters'}       # Columns without questions and asked questions
        st.session_state.df_factual = load_method_data()
        st.session_state.df_factual_original = load_method_data()
        st.session_state.df_questions = load_question_data()
        st.session_state.df_approaches = load_approaches_data()
        st.session_state.current_index = 0
        st.session_state.decision_log = pd.DataFrame(columns=['Attribute', 'Answer', 'Removed_Methods', 'Remaining_methods'])  # Stores decisions made by user
        st.session_state.number_dc = len(find_decision_classes(st.session_state.df_factual, st.session_state.excluded_columns)[0])
        st.session_state.db_status = "Running"
        st.session_state.reran = False

    # Initialize everything after rerunning
    if st.session_state.reran == True:
        reran_df_factual, reran_decision_log = create_old_dataframe(
                        st.session_state.decision_log, 
                        st.session_state.df_factual_original, 
                        st.session_state.reran_attribute)
        st.session_state.excluded_columns = {'Name', 'Class', 'Zetoro', 'Tunable Paramaters'}       # Columns without questions and asked questions
        st.session_state.df_factual = reran_df_factual
        st.session_state.df_factual_original = load_method_data()
        st.session_state.df_questions = load_question_data()
        st.session_state.df_approaches = load_approaches_data()
        st.session_state.current_index = 0
        st.session_state.decision_log = reran_decision_log  # Stores decisions made by user
        st.session_state.number_dc = len(find_decision_classes(st.session_state.df_factual, st.session_state.excluded_columns)[0])
        st.session_state.db_status = "Running"
        st.session_state.result_graph = None
        st.session_state.reran = False

    with col1:
        st.session_state.number_dc = len(find_decision_classes(st.session_state.df_factual, st.session_state.excluded_columns)[0])
        # If 5 columns are left over, all questions are exhausted -> stop
        if len(st.session_state.df_factual.columns) == len(st.session_state.excluded_columns):
            st.session_state.db_status = "Terminated"
            st.write(':gray[No more questions to ask]')
            st.write(':gray[Recommended method(s): ]')
            # Create original dataframe to display full methods
            st.session_state.methods_over = st.session_state.df_factual['Name'].to_list()
            st.session_state.original_df = load_method_data()
            st.session_state.maskEnd = st.session_state.original_df['Name'].isin(st.session_state.methods_over)
            # Add select column to the methods and rearrange columns
            st.session_state.original_df["Select"] = False
            cols = st.session_state.original_df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            st.session_state.original_df = st.session_state.original_df[cols]
            # Display results and select method
            edited_df = st.data_editor(st.session_state.original_df[st.session_state.maskEnd], use_container_width=True)
            selected_rows = edited_df[edited_df["Select"]]
            # Display decision tree
            st.write(':gray[Decision log: ]')
            st.write(st.session_state.decision_log)

            requirement_only_list = find_additional_questions(st.session_state.df_approaches)
            potential_approaches = find_no_req(st.session_state.df_approaches)
            option = st.multiselect(
                ":gray[Select all possibilities that apply to your problem]",
                requirement_only_list,
            )
            if st.button('Apply'):
                df_approaches_modified = remove_misreq(st.session_state.df_approaches, st.session_state.original_df, selected_rows['Name'].values[0])
                df_approaches_final = remove_add_misreq(df_approaches_modified, option)
                st.write(":gray[Potetential evaluation approaches]")
                st.write(potential_approaches)
                st.write(":gray[Potetential evaluation approaches with req]")
                st.write(df_approaches_final)

        # If there are more than 5 columns left over, ask a remaining questions
        else:
            if st.session_state.number_dc > 1:
                # Create question
                st.session_state.answ = find_question(st.session_state.df_factual, st.session_state.df_questions, st.session_state.excluded_columns)
                st.session_state.answ_options = st.session_state.answ[4].split(',')
                # Ask question
                st.title(f"**"+st.session_state.answ[1]+"**")
                st.write(f':gray[Possible amount of answers: {str(st.session_state.answ[3])}]')
                # Create answer columns
                st.session_state.number_columns = len(st.session_state.answ_options)
                st.session_state.cols = st.columns(st.session_state.number_columns)
                st.session_state.current_index += 1

                # If only 1 answer is allowed and needed
                if str(st.session_state.answ[3]) == '1':
                    for i, col in enumerate(st.session_state.cols):     # For each solution, create button
                        with col:
                            st.write(f':gray[{st.session_state.answ_options[i]}]') 
                            st.image(f"icons_DB/"+st.session_state.answ[0]+'/'+st.session_state.answ[2].split(',')[i]+'.png', caption=f"testpicture{i+1}")
                            if st.button(f"Answer {i+1}", key=f"btn_{i}"):      # Selected answer
                                # Processs answer
                                st.session_state.df_factual, st.session_state.decision_log = process_answer_updated(
                                    st.session_state.answ[0],
                                    st.session_state.answ[2].split(',')[i], 
                                    st.session_state.df_factual, 
                                    st.session_state.df_questions,
                                    st.session_state.decision_log,
                                    st.session_state.current_index)             
                                st.rerun()
                # If multiple or no answer is allowed
                else:
                    # Create a list for selected answers
                    if "current_answer_selection" not in st.session_state:
                        st.session_state.current_answer_selection = []
                    for i, col in enumerate(st.session_state.cols):
                        with col:
                            # Create answer options
                            option = st.session_state.answ[2].split(',')[i]
                            st.write(f':gray[{st.session_state.answ_options[i]}]')
                            st.image(f"icons_DB/{st.session_state.answ[0]}/{option}.png",caption=f"testpicture{i+1}")
                            # Create buttons for the answers
                            if st.button(f"Answer {i+1}", key=f"btn_{i}"):
                                # Removes answer if it is already there
                                if option in st.session_state.current_answer_selection:
                                    st.session_state.current_answer_selection.remove(option)
                                # Adds answer if not
                                else:
                                    st.session_state.current_answer_selection.append(option)

                    if st.button("Submit answers", key="submit_answers"):      # Finish selecting answers
                        # Processs answer
                        st.session_state.df_factual, st.session_state.decision_log = process_answer_updated(
                            st.session_state.answ[0],
                            st.session_state.current_answer_selection,
                            st.session_state.df_factual,
                            st.session_state.df_questions,
                            st.session_state.decision_log,
                            st.session_state.current_index
                            )
                        # Clear selections for next question
                        st.session_state.current_answer_selection = []
                        st.rerun()
                    st.write(f":gray[Selected answers:{st.session_state.current_answer_selection}]")
            else:
                if st.session_state.number_dc == 1:
                    st.session_state.db_status = "Terminated"
                    st.write(':gray[Recommended method(s): ]')
                    # Create original dataframe to display full methods
                    st.session_state.methods_over = st.session_state.df_factual['Name'].to_list()
                    st.session_state.original_df = load_method_data()
                    st.session_state.maskEnd = st.session_state.original_df['Name'].isin(st.session_state.methods_over)
                    
                    # Display result
                    selected_row = st.session_state.original_df[st.session_state.maskEnd]
                    edited_df = st.write(selected_row)
                    # Display decision tree
                    st.write(':gray[Decision log: ]')
                    st.write(st.session_state.decision_log)      
                    # Create and find metrics
                    requirement_only_list = find_additional_questions(st.session_state.df_approaches)
                    potential_approaches = find_no_req(st.session_state.df_approaches)
                    option = st.multiselect(
                        ":gray[Select all possibilities that apply to your problem]",
                        requirement_only_list,
                    )
                    # Additional requirements for metrics
                    if st.button('Apply'):
                        df_approaches_modified = remove_misreq(st.session_state.df_approaches, st.session_state.original_df, selected_row['Name'].values[0])
                        df_approaches_final = remove_add_misreq(df_approaches_modified, option)
                        st.write(":gray[Potetential evaluation approaches]")
                        st.write(potential_approaches)
                        st.write(":gray[Potetential evaluation approaches with req]")
                        st.write(df_approaches_final)

                else:
                    st.session_state.db_status = "Terminated"
                    st.write(':gray[No methods found with these requirements]')
                    st.write(':gray[Decision log: ]')
                    st.write(st.session_state.decision_log)


    with col2:    

        if st.session_state.db_status == "Running":
            st.session_state.small_graph = graphviz.Digraph('Q-A', node_attr={'shape': 'plaintext'})
            st.session_state.small_graph = create_single_question(st.session_state.answ, st.session_state.df_factual, st.session_state.df_questions)
            st.graphviz_chart(st.session_state.small_graph)
        
    if st.session_state.db_status == "Terminated":
        with st.popover(":blue[Show decision tree]", use_container_width=True):
            # Backtracking
            bt_option = st.selectbox(
                "Select attribute to backtrack to",
                st.session_state.decision_log['Attribute'],
            )
            if st.button('Apply', key='Backtrack'):
                st.session_state.reran_attribute = bt_option
                st.session_state.reran = True
                st.rerun()

            # Creates bactracking graph
            st.session_state.df_factual_graph, st.session_state.decision_log_graph = preprocess_data_graph(st.session_state.df_questions.copy(),st.session_state.df_factual_original.copy(), st.session_state.decision_log.copy())
            st.session_state.result_graph = graphviz.Digraph('Q-A', node_attr={'shape': 'plaintext'})
            st.session_state.result_graph = create_question_nodes(st.session_state.result_graph, st.session_state.decision_log_graph)
            st.session_state.result_graph = create_answer_nodes(st.session_state.result_graph, st.session_state.decision_log_graph, st.session_state.df_factual_graph)
            # Graph styling
            st.session_state.result_graph.attr(rankdir='TB', nodesep='0.5', ranksep='0.7')
            st.session_state.result_graph.attr('graph', splines='polyline')
            st.session_state.result_graph = align_graph(st.session_state.result_graph, st.session_state.decision_log_graph)
            st.session_state.result_graph.attr(rankdir='LR')        
            st.graphviz_chart(st.session_state.result_graph)



    if st.session_state.db_status != "Terminated" and len(st.session_state.decision_log['Attribute']) > 0:
        with st.popover(":blue[Show decision tree]", use_container_width=True):
            # Backtracking
            st.session_state.bt_option = st.selectbox(
                "Select attribute to backtrack to",
                st.session_state.decision_log['Attribute'],
            )
            if st.button('Backtrack', key='Backtrack'):
                st.session_state.reran_attribute = st.session_state.bt_option
                st.session_state.reran = True
                st.rerun()
            
            # Create backtracking graph
            st.session_state.df_factual_graph, st.session_state.decision_log_graph = preprocess_data_graph(st.session_state.df_questions.copy(),st.session_state.df_factual_original.copy(), st.session_state.decision_log.copy())
            st.session_state.result_graph = graphviz.Digraph('Q-A', node_attr={'shape': 'plaintext'})
            st.session_state.result_graph = create_question_nodes(st.session_state.result_graph, st.session_state.decision_log_graph)
            st.session_state.result_graph = create_answer_nodes(st.session_state.result_graph, st.session_state.decision_log_graph, st.session_state.df_factual_graph)
            # Graph styling
            st.session_state.result_graph.attr(rankdir='TB', nodesep='0.5', ranksep='1')
            st.session_state.result_graph.attr('graph', splines='polyline')
            st.session_state.result_graph = align_graph(st.session_state.result_graph, st.session_state.decision_log_graph)
            st.session_state.result_graph.attr(rankdir='LR')

            st.graphviz_chart(st.session_state.result_graph)
        

