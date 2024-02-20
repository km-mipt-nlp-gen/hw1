class ChatRepository:
    def __init__(self, chat_msg_history, target_char_preprocessed_qa_pairs, target_char_preprocessed_answers,
                 bi_encoder_model, cross_encoder_model, chat_service_accelerator):
        self._chat_msg_history = chat_msg_history
        self._preprocessed_questions_answers_embeddings = chat_service_accelerator.preprocess_training_data_embeddings(
            target_char_preprocessed_qa_pairs)
        self._preprocessed_answers_embeddings = chat_service_accelerator.preprocess_answers_embeddings(
            target_char_preprocessed_answers)
        self._bi_encoder_model = bi_encoder_model
        self._cross_encoder_model = cross_encoder_model
        self._target_char_preprocessed_answers = target_char_preprocessed_answers
        self._target_char_preprocessed_qa_pairs = target_char_preprocessed_qa_pairs
        self._preprocessed_questions_answers_embeddings_faiss_index = chat_service_accelerator.create_faiss_index(
            self._preprocessed_questions_answers_embeddings)
        self._preprocessed_questions_answers_embeddings_faiss_psa_index = chat_service_accelerator.create_faiss_index_psa(
            self._preprocessed_questions_answers_embeddings)

    @property
    def chat_msg_history(self):
        return self._chat_msg_history

    @chat_msg_history.setter
    def chat_msg_history(self, value):
        self._chat_msg_history = value

    @property
    def preprocessed_questions_answers_embeddings(self):
        return self._preprocessed_questions_answers_embeddings

    @property
    def preprocessed_answers_embeddings(self):
        return self._preprocessed_answers_embeddings

    @property
    def bi_encoder_model(self):
        return self._bi_encoder_model

    @property
    def cross_encoder_model(self):
        return self._cross_encoder_model

    @property
    def target_char_answers(self):
        return self._target_char_preprocessed_answers

    @property
    def target_char_questions_and_answers(self):
        return self._target_char_preprocessed_qa_pairs

    @property
    def preprocessed_questions_answers_embeddings_faiss_index(self):
        return self._preprocessed_questions_answers_embeddings_faiss_index

    @property
    def preprocessed_questions_answers_embeddings_faiss_psa_index(self):
        return self._preprocessed_questions_answers_embeddings_faiss_psa_index