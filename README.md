# Word-Embedding-technique-using-CBOW---Skipgram
# Aim Of the Experiement:
The aim of this experiment is to learn and analyze word embeddings from a textual document using neural language models, specifically Continuous Bag-of-Words (CBOW) and Skip-gram, by investigating how context window size and embedding dimensionality influence the quality of the learned vector representations and comparing the semantic properties captured by each model.

# Mathematical Intuition:
Both CBOW and Skip-gram learn word embeddings by maximizing the likelihood of observed word–context co-occurrences within a fixed context window, thereby embedding words into a continuous vector space. In CBOW, the context word vectors surrounding a target word are averaged to form a single representation, and the model maximizes the probability of the target word given this averaged context vector, which mathematically encourages frequent contextual patterns to cluster together. In Skip-gram, a single target word vector is used to predict multiple surrounding context words, maximizing the joint probability of context words conditioned on the target; this results in stronger gradient updates for each word–context pair, particularly benefiting rare words. To avoid the computational cost of the full softmax over the vocabulary, both models use negative sampling, which reformulates the objective as a binary classification problem that increases the dot product between true word–context pairs while decreasing it for randomly sampled noise pairs, shaping the embedding space such that semantically related words lie closer together in terms of vector similarity.

CBOW Objective
Given a target word w_tand its context \left\{w_{t-w},\ldots,w_{t-1},w_{t+1},\ldots,w_{t+w}\right\}, the context representation is computed as:
<img width="165" height="65" alt="image" src="https://github.com/user-attachments/assets/321a25f9-d487-4d21-86cf-c3b649f84cc3" />

The probability of the target word is:
<img width="261" height="62" alt="image" src="https://github.com/user-attachments/assets/68da2901-45ce-431b-b488-2c94530efe84" />

Skip-gram Objective
Given a center word w_t, the model predicts each context word w_{t+j}independently:
<img width="668" height="358" alt="image" src="https://github.com/user-attachments/assets/8dfa1335-ddac-476f-9392-6e8ba22345d2" />
<img width="757" height="153" alt="image" src="https://github.com/user-attachments/assets/a0dcc670-7fa9-4768-96d1-03614981dda0" />

# Task Overview and flowchart:
<img width="940" height="466" alt="image" src="https://github.com/user-attachments/assets/ec0daeca-c243-4a79-939d-166226736ea3" />

# Model Evaluation and Result evaluation:
The performance of the CBOW and Skip-gram models was evaluated quantitatively using training loss values and cosine similarity scores derived from the learned embeddings. For the CBOW model, the average training loss decreased from approximately 10.38 in the first epoch to 8.40 by the second epoch, continuing to decline steadily over subsequent epochs, indicating fast and stable convergence. The Skip-gram model showed a similar trend, with the loss reducing from about 9.99 to 8.60 over the same initial epochs, albeit with a slightly higher computational cost due to the larger number of training pairs.
In terms of embedding quality, cosine similarity measurements revealed clearer semantic relationships in the Skip-gram embeddings. For example, the cosine similarity between the semantically related words legal and refund was approximately 0.25 in the Skip-gram model, compared to only 0.06 in the CBOW model. Nearest-neighbor evaluations further supported this observation: Skip-gram consistently retrieved more semantically coherent neighbors, whereas CBOW produced smoother but less discriminative associations. Overall, the numerical results demonstrate that while CBOW achieves faster convergence with lower computational overhead, Skip-gram yields embeddings with stronger semantic structure and higher similarity scores for related word pairs.

# Conclusion:
In this experiment, word embeddings were successfully learned from a textual document using the Continuous Bag-of-Words (CBOW) and Skip-gram neural language models. Both models demonstrated stable training behavior, with a consistent reduction in loss values across epochs, confirming effective optimization using negative sampling. Quantitative evaluation showed that CBOW converged faster and produced smoother representations, while Skip-gram achieved higher cosine similarity scores and more semantically meaningful nearest neighbors, particularly for related and less frequent words. These numerical and qualitative results align with the theoretical expectations of the two models, leading to the conclusion that CBOW is better suited for efficient training on frequent patterns, whereas Skip-gram is more effective for capturing richer semantic relationships in word embedding spaces.



