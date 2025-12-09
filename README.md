**Abstract**

The challenge we are trying to solve with our project is that it is difficult to create effective study materials based on class content and human preferences, since traditional word-based methods don’t have a good semantic understanding and flexibility across various different flashcard sets. To address this, we created a pipeline that loads, parses, and preprocesses Quizlet flashcards, classifies them using rule-based heuristics and transformer-based zero-shot learning, and embeds them into a vector database for semantic search. We then use interactive notebooks for incremental updates, managing metadata, and querying natural language. This workflow allows for semantic retrieval, flashcard classification, and retrieval-augmented generation (RAG), which we use to create an adaptable and scalable AI-powered study tool. (wrote this based on the points above, feel free to add to this)

**Overview**

Current AI tools like ChatGPT are effective at generating summaries and notes but fall short when creating flashcards that structure content in the most effective way and is exam-based. They often generate dictionary‑like lists that aren’t really effective and struggle when multiple correct answers exist or capturing the larger picture of how concepts connect across procedures or contexts. This results in flashcards that emphasize just recall instead of meaningful structure or contextual understanding. The motivation behind this project is to create an AI system that can contextualize real examples from a student’s own classes, like past exams, quizzes, and assignments, and transform them, along with other provided documentations, into study materials that reflect the types of questions students are likely to see on their next exam.

This problem is particularly interesting because it’s relevant to many students, who often face the tedious and time‑consuming task of creating flashcards manually when studying. Without a universal, user‑friendly solution, many learners lose motivation and so flashcards are not a good study option. By using AI to generate tailored flashcards that’s based on real exam questions, the project addresses a need among students. Since universities vary in how they teach and test the same topics, the ability to adapt flashcards to specific courses and question formats helps students receive personalized, context‑aware study materials. Overall, our project could support more effective learning, reduce study demotivation, and allow access to high‑quality exam preparation tools for everyone.

Our approach uses an AI pipeline with retrieval‑augmented generation (RAG). The workflow begins with a preprocessing step, where course materials, such as PDFs, are parsed and transformed into JSON documents suitable for retrieval. Next, a formatting step uses historical data (like past exams, quizzes, assignments) to get the types of questions typically asked in a given class. This format is stored per topic (class subject), which effects the generated flashcard. In the generation step, user‑provided documents are combined with top‑k relevant contextual documents retrieved via RAG. The backend prompts a large language model with this curated context to produce structured JSON flashcards, which are then stored in a database and displayed through a frontend interface. Each topic has its own page, so students can have historical question types add new topic materials.

The rationale for using RAG instead of fine‑tuning a baseline language model is that it is more efficient and accurate. RAG reduces computational cost while increasing the output quality by basing generation on the most relevant course materials. This reduces hallucinations and ensures that flashcards reflect actual exam topics instead of generic question types. While prior work, like Quizlet Premium, has flashcard generation abilities, this project differs by using user‑specific historical data to consider the unique exam styles of individual classes. The result is a more curated and personalized study experience that adapts dynamically to new topics and materials, and is based off of a more diverse set of high-quality flashcards.

The key components of the projects includes preprocessing pipelines for document parsing, a formatting section for getting the exam styles, a RAG‑based retrieval process, and a frontend interface for organizing flashcards for each user. We found that our project can classify flashcards, embed them with metadata, and support semantic search and filtering. We can now query flashcards using natural language, retrieve relevant examples, and study with materials that is based on their actual coursework. There are some limitations, such as it’s reliance on the quality of user‑provided historical data, so bad or incomplete exam samples might generated formats less accurate. Also, we need ongoing updates to the vector store as the user puts in new materials.. However, the workflow is pretty scalable and adaptable, and makes the difference between generic flashcard generation and personalized exam preparation.

**Approach** 

The overall approach is a pipeline that processes flashcards from different formats, classifies them, and makes them searchable with semantic methods. A custom loader parses and cleans the data, adds metadata, and automatically classifies flashcard types. Classification is done with a transformer‑based zero‑shot model, which does not require labeled training data. When the model output is weak or unavailable, rule‑based heuristics using keywords and patterns act as a fallback. Flashcards are then converted into vector embeddings with a sentence transformer model and stored in Chroma, a persistent vector database. This setup allows fast similarity search and retrieval, with results filtered by metadata and queried using natural language.

The design assumes that flashcard data is structured with clear “term” and “definition” fields, and that most files follow JSON or tab‑delimited text formats. File extensions and simple content checks are used to detect formats, which may not cover corrupted or unusual cases. The classification pipeline assumes the model generalizes well to educational content, while the fallback relies on predictable keywords and patterns. These choices keep the system lightweight and flexible but also introduce risks if the input data does not match expected structures.

Limitations include possible misclassification of ambiguous or poorly worded flashcards, brittle rule‑based fallbacks, and parsing errors with non‑standard file formats. The embedding and classification models are not fine‑tuned for Quizlet or educational flashcards, which may reduce semantic accuracy. Large datasets can be slow or memory‑intensive to process, and missing dependencies reduce functionality to basic rule‑based methods. Because metadata accuracy is important, errors in preprocessing can affect downstream search and filtering. Finally, the system does not include a feedback loop, so mistakes in classification or retrieval persist unless manually corrected.

**Experiments**

The dataset for this project was collected from popular Quizlet flashcard sets, which reflect real-world, user-generated study materials. To ensure balance across different flashcard types, we also added a small hand-crafted set with 5–10 examples of each type. The content spans a wide range of subjects and includes formats such as term-definition, multiple choice, fill-in-the-blank, true/false, list stages, example-to-concept, and question-to-answer. Each flashcard is structured with a “term” and “definition” field, and files are stored in JSON or tab-delimited text formats.

The Quizlet sets vary from dozens to hundreds of cards, and combined with the hand-crafted additions, the dataset typically contains several hundred flashcards. Metadata is included for each card, such as source file, card number, and automatically classified flashcard type. The hand-crafted set ensures that all flashcard types are represented (for a balanced dataset), so that we can have a fair evaluation of classification and retrieval methods. Overall, the dataset considers effective study habits while also providing curated balance for testing.

The implementation uses a mix of transformer models and rule-based methods to classify and embed flashcards. For classification, the system runs the \`MoritzLaurer/deberta-v3-large-zeroshot-v2.0\` model through Hugging Face with an \`LLMClient\` wrapper. This assigns flashcard types such as term-definition, multiple choice, or fill-in-the-blank using natural language prompts and candidate labels. Batch classification is supported to improve speed. If the model is unavailable or uncertain, rule-based heuristics using keyword and pattern matching are applied, and strong matches can override model predictions. For semantic search, flashcards are embedded with \`sentence-transformers/all-MiniLM-L6-v2\` and stored in Chroma.

The setup uses Python 3.11 with libraries including Hugging Face Transformers, LangChain Core, Chroma, and standard Python packages. Experiments are run in VS Code and Jupyter Notebooks without the need for GPU. Parameters include a confidence threshold of 0.5 for model predictions, canonical labels for flashcard types, and optional cues that can be prepended to input text. Flashcards are loaded with a custom \`QuizletLoader\`, classified in batches, and enriched with metadata before being embedded and saved. Preprocessing and querying are managed interactively in notebooks, allowing users to update the vector store and perform semantic searches with natural language queries. 

Our experiment’s model architecture runs a transformer-based encoder-decoder model that applies attention to compare input text with candidate labels and assign a flashcard type. This is done in a zero-shot way using pre-trained weights without fine-tuning for classification. For embeddings, it uses a transformer-based way with multiple layers and applies mean pooling over token embeddings to create a fixed-size vector for each flashcard. These vectors are stored in Chroma. Finally, flashcard generation is handled by an API client that connects to a large language model. This model produces new flashcards using retrieval-augmented generation with context pulled from the vector database.

**Results**

The experiments showed that semantic search worked well, retrieving flashcards that matched natural language queries even when the wording is different from the original card. This means we can find cards by concept, topic, or question instead of relying on exact keyword matches. The zero-shot classification model was able to assign flashcard types accurately for most cases, especially when clear cues were there, while rule-based heuristics provided a good fallback for structured formats like multiple choice or true/false. Metadata filtering also proved useful, since it lets us narrow results for more focused study materials..  

One of the most interesting findings was that zero-shot models generalized well without fine-tuning, handling diverse flashcard types as long as the input text was clear. Embedding-based search improved retrieval by capturing conceptual similarity, not just keyword overlap, which made results more effective. Balanced hand-crafted sets helped with edge cases and better classification and retrieval results. Parameter choices also helped with better outcomes. The embedding model was chosen for efficiency and the classifier had strong zero-shot performance. A set confidence threshold reduced misclassification, with rule-based fallbacks for reliability. Treating each flashcard as an individual chunk and limiting query results kept things manageable.

**Discussion**

Compared to existing approaches, our results show improvements in how flashcards can be organized and searched. Common apps like Quizlet mostly rely on keyword search and manual organization, which is bad with flexibility and context. ChatGPT can generate notes but struggles to produce exam-style questions or connect concepts. On the other hand, our project uses transformer-based zero-shot classification and semantic embeddings to assign flashcard types and retrieve cards by meaning instead of just exact wording. Through our project, we showed how semantic search and classification outperform simple keyword or rule-based systems from other approaches.

At the same time, the experiments did have some challenges. The dataset was relatively small compared to real-world usage, which limits its strength. Zero-shot models were not fine-tuned for educational flashcards, so domain-specific cards were sometimes misclassified. The project also focused more on qualitative outcomes than quantitative, so there’s room for more testing. Future improvements could be adding user feedback for better classifications, fine-tuning models on domain-specific data, and maybe building personalized study recommendations based on user progress. We would also make a better learning experience by changing up the flashcard order and type based on the user’s skill level. Adding these features would help increase the educational impact and make the project more useful for students on a larger scale.

**Conclusion**

Through the experiments in our project, we built an AI-powered study tool that can classify, organize, and search flashcards in a smarter way than traditional keyword-based approaches in common apps for flashcard generation. By combining zero-shot classification, rule-based heuristics, and semantic embeddings stored in a vector database, we retrieve relevant flashcards with natural language queries and filter them by metadata. We also showed that zero-shot models can assign flashcard types without labeled training data. Overall, our working pipeline turns real class materials into structured, searchable flashcards, making a personalized and more effective study experience.

**APA-Style References** 

**1\.** Amazon Web Services. (n.d.). *What is retrieval-augmented generation (RAG)?* [https://aws.amazon.com/what-is/retrieval-augmented-generation/](https://aws.amazon.com/what-is/retrieval-augmented-generation/)

**2\.** LangChain. (n.d.). *Retrieval.* [https://docs.langchain.com/oss/python/langchain/retrieval](https://docs.langchain.com/oss/python/langchain/retrieval)

**3\.** LangChain. (n.d.). *Document loaders by category.*  
 [https://docs.langchain.com/oss/python/integrations/document\_loaders\#by-category](https://docs.langchain.com/oss/python/integrations/document_loaders#by-category)

**4\.** LangChain. (n.d.). *Use these docs.* [https://docs.langchain.com/use-these-docs](https://docs.langchain.com/use-these-docs)

**5\.** Chrome Web Store. (n.d.). *Quizlet Exporter – Export flashcards to CSV.*  
 [https://chromewebstore.google.com/detail/quizlet-exporter-export-f/lkoaedeomnobdibjfdfggjhoiiabpgkl](https://chromewebstore.google.com/detail/quizlet-exporter-export-f/lkoaedeomnobdibjfdfggjhoiiabpgkl)

**6\.** Quizlet. (n.d.). *Artificial intelligence flashcards (Set 12266018).*  
 [https://quizlet.com/12266018/artificial-intelligence-flash-cards/](https://quizlet.com/12266018/artificial-intelligence-flash-cards/)

**7\.** Quizlet. (n.d.). *Artificial intelligence flashcards (Set 22838468).*  
 [https://quizlet.com/22838468/artificial-intelligence-flash-cards/](https://quizlet.com/22838468/artificial-intelligence-flash-cards/)

**8\.** Quizlet. (n.d.). *Artificial intelligence flashcards (Set 831997215).*  
 [https://quizlet.com/831997215/artificial-intelligence-flash-cards/](https://quizlet.com/831997215/artificial-intelligence-flash-cards/)

**9\.** Quizlet. (n.d.). *Artificial intelligence flashcards (Set 885285622).*  
 [https://quizlet.com/885285622/artificial-intelligence-flash-cards/](https://quizlet.com/885285622/artificial-intelligence-flash-cards/)

**10\.** Quizlet. (n.d.). *Artificial intelligence flashcards (Set 324526330).*  
 [https://quizlet.com/324526330/artificial-intelligence-flash-cards/](https://quizlet.com/324526330/artificial-intelligence-flash-cards/)

**11\.** LangChain. (n.d.). *Chroma vector store integration.*  
 [https://docs.langchain.com/oss/python/integrations/vectorstores/chroma](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)

**12\.** MoritzLaurer. (n.d.). *deberta-v3-large-zeroshot-v2.0* \[Model\]. Hugging Face.  
 [https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0)

**13\.** Data Science Dojo. (n.d.). *How to build a rule-based chatbot in Python.*  
 [https://datasciencedojo.com/blog/rule-based-chatbot-in-python/](https://datasciencedojo.com/blog/rule-based-chatbot-in-python/)

**Evaluation** 

We evaluate the effectiveness of our model by comparing the quality of generated flashcards based on the same class materials and providing homework question structure from our AI model vs chatgpt's output. We found that many of our flashcards cover core foundational concepts and probes for a deeper understanding of topics by assessing knowledge across different concepts. For example, the question on Model-Based vs. Model-Free Reinforcement Learning reflects something a real exam might ask in order to test understanding of connecting topics in reinforcement learning. On the other hand, an example of a question ChatGPT generated asks for an example but only provides a portion of what could be the right answer.

Another example of the difference between our model output v chatgpt’s is the flashcards generated on our class’s language models content. For instance,  ChatGPT gave a flashcard asking for a definition of the temperature parameter, which is a relevant topic, but compared to language model fundamentals, NLP sampling temperature is usually a smaller subtopic and less foundational theoretically.  For this topic, a flashcard that our model made focuses on showing applied understanding through prompting for examples. What makes this output different is the diverse range of answers it provides so that the answer is both informative and covers most answers that can be provided.

Our evaluation shows that our model produces flashcards that more closely align with the conceptual depth and exam-style reasoning expected in courses. Compared to ChatGPT’s outputs, which often focus on surface-level definitions or partial answers, our model consistently generates prompts that target fundamental principles and require students to integrate ideas across topics. This was especially evident in the randomly selected flashcard examples that were generated from the same provided content, where our flashcards captured core theoretical constructs and encouraged applied reasoning, while the other model emphasized narrower or less central subtopics. Overall, the results show that our workflow provides study materials that better reflect the structure, rigor, and expectations of real exam questions.

Reference documents: hw3, hw4, hw5  
Our example output on MDP:

- Term: Model-Based vs. Model-Free RL  
- Definition: Model-based methods use a transition model to plan, while model-free methods learn directly from experience. Model-based approaches can be more sample-efficient but require accurate environment knowledge.  
- Flashcard type: term\_definition

Chatgpt output for week 11 notes:

- Term: Give an example where the Markov property is violated.  
- Definition: A robot whose battery level depends on long-term past movement, not just the current state; or predicting stock prices where historical momentum matters.  
- Flashcard type: example\_to\_concept

Our example output on on Language model:

- Term: NLP Task Examples  
- Definition: Language models are applied to tasks like sentiment analysis (classifying text polarity), toxicity detection, machine translation, named entity recognition, spam detection, grammatical error correction, topic modeling, text generation, information retrieval, summarization, and question answering.  
- Flashcard type: list\_stages

Chatgpt output for week 5+6 notes

- Term: What is the temperature parameter T?  
- Definition: A value controlling randomness in sampling from the LM.  
- Type: question\_answer

[demo](https://drive.google.com/drive/folders/1f1plEyhdOFLPqLHR0pAvOaMZzoK2OTbs?usp=sharing)  
\* Note on frontend: generated by Claude AI for demonstration purposes