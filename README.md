# NLP_Assignment3

Dialogue Agents: retrieve next best uttereance The topic of the last exercise will be around dialogue system, in particular for non goal-oriented (chit-chat) dialogues. For that we will use the data from the ConvAI challenge, to be downloaded from here . 
For each utterance, you will be given N+1 possible answers of which you have to pick out the correct one. The format of the training will be: utterance TAB correctAnswer TAB distractor1 | distractor2 | ... | distractorN-1, and of testing: utterance TAB possibleAnswer1 | ... | possibleAnswerN+1.

We also built a generator function to generate a response instead of selecting among multiple distractiors
