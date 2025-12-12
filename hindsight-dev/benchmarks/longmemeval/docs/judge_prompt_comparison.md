# LongMemEval Judge Prompt Comparison: Original Paper vs Hindsight

## 1. `single-session-user`, `single-session-assistant`, `multi-session`

| Original Paper | Hindsight |
|----------------|-----------|
| I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. | Evaluate if the model response contains the correct answer to the question. |
| | I will give you a question, a correct answer, and a response from a model. Please set correct=true if the response contains the correct answer. Otherwise, set correct=no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also set correct=true. If the response only contains a subset of the information required by the answer, set correct=false |
| Question: {question} | Question: {question} |
| Correct Answer: {answer} | Correct Answer: {correct_answer} |
| Model Response: {response} | Model Response: {predicted_answer} |
| Is the model response correct? Answer yes or no only. | Evaluation criteria: |
| | - Set correct=true if the response contains the correct answer |
| | - Set correct=true if the response is equivalent to the correct answer or contains intermediate steps |
| | - Set correct=false if the response is incorrect or missing key information |
| | Provide your evaluation as JSON with: |
| | - reasoning: One sentence explanation |
| | - correct: true or false |

---

## 2. `temporal-reasoning`

| Original Paper | Hindsight |
|----------------|-----------|
| I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. | I will give you a question, a correct answer, and a response from a model. Please set correct=true if the response contains the correct answer. Otherwise, set correct=false. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also set correct=true. If the response only contains a subset of the information required by the answer, answer correct=false. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. |
| Question: {question} | Question: {question} |
| Correct Answer: {answer} | Gold answer: {correct_answer} |
| Model Response: {response} | Generated answer: {predicted_answer} |
| Is the model response correct? Answer yes or no only. | First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred. If it's correct, set correct=true. |

---

## 3. `knowledge-update`

| Original Paper | Hindsight |
|----------------|-----------|
| I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer. | I will give you a question, a correct answer, and a response from a model. Please set correct=true if the response contains the correct answer. Otherwise, set correct=false. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer. |
| Question: {question} | Question: {question} |
| Correct Answer: {answer} | Gold answer: {correct_answer} |
| Model Response: {response} | Generated answer: {predicted_answer} |
| Is the model response correct? Answer yes or no only. | First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred. If it's correct, set correct=true. |

---

## 4. `single-session-preference`

| Original Paper | Hindsight |
|----------------|-----------|
| I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly. | I will give you a question, a answer for desired personalized response, and a response from a model. Please set correct=true if the response satisfies the desired response. Otherwise, set correct=false. The model does not need to reflect all the points in the desired response. The response is correct as long as it recalls and utilizes the user's personal information correctly. |
| Question: {question} | Question: {question} |
| Rubric: {rubric} | Gold answer: {correct_answer} |
| Model Response: {response} | Generated answer: {predicted_answer} |
| Is the model response correct? Answer yes or no only. | First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred. If it's correct, set correct=true. |

---

## 5. `unanswerable` (abstention)

| Original Paper | Hindsight |
|----------------|-----------|
| I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not. | *Not implemented* |
| Question: {question} | |
| Explanation: {explanation} | |
| Model Response: {response} | |
| Does the model correctly identify the question as unanswerable? Answer yes or no only. | |

---

## 6. Default (fallback for unknown categories)

| Original Paper | Hindsight |
|----------------|-----------|
| *No default - all categories have specific prompts* | Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG. |
| | The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example: Question: Do you remember what I got the last time I went to Hawaii? Gold answer: A shell necklace The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. |
| | For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date. |
| | There's an edge case where the actual answer can't be found in the data and in that case the gold answer will say so (e.g. 'You did not mention this information.'); if the generated answer says that it cannot be answered or it doesn't know all the details, it should be counted as CORRECT. |
| | Question: {question} |
| | Gold answer: {correct_answer} |
| | Generated answer: {predicted_answer} |
| | First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred. If it's correct, set correct=true. |
