from PySimpleAutomata import DFA, automata_IO

dfa_example = automata_IO.dfa_json_importer("exampleBIS.json")

new_dfa=DFA.dfa_reachable(dfa_example)

automata_IO.dfa_to_dot(new_dfa, 'outputttt-name', './')