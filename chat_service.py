import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


class ChatService:
    def __init__(self, chat_msgs_history, chat_repository, constants, chat_util):
        self._chat_msgs_history = chat_msgs_history
        self.repository = chat_repository
        self.constants = constants
        self.chat_util = chat_util

    @property
    def chat_msgs_history(self):
        return self.repository.chat_msgs_history

    @property
    def preprocessed_questions_answers_embeddings(self):
        return self.repository.preprocessed_questions_answers_embeddings

    @property
    def preprocessed_answers_embeddings(self):
        return self.repository.preprocessed_answers_embeddings

    @property
    def bi_encoder_model(self):
        return self.repository.bi_encoder_model

    @bi_encoder_model.setter
    def bi_encoder_model(self, value):
        self.repository.bi_encoder_model = value

    @property
    def cross_encoder_model(self):
        return self.repository.cross_encoder_model

    @cross_encoder_model.setter
    def cross_encoder_model(self, value):
        self.repository.cross_encoder_model = value

    @property
    def cross_encoder_chunk_size(self):
        return self.repository.cross_encoder_chunk_size

    @property
    def target_char_answers(self):
        return self.repository.target_char_answers

    @property
    def target_char_questions_and_answers(self):
        return self.repository.target_char_questions_and_answers

    @property
    def preprocessed_questions_answers_embeddings_faiss_index(self):
        return self.repository.preprocessed_questions_answers_embeddings_faiss_index

    def get_query_answers_embeddings_bi_encoder(self, tokenizer, question):
        self.bi_encoder_model = self.bi_encoder_model.to(self.constants.device)
        question_tokens = tokenizer(question, return_tensors="pt", padding='max_length', truncation=True,
                                    max_length=128).to(self.constants.device)
        with torch.no_grad():
            question_embeds = self.bi_encoder_model.bert_model(input_ids=question_tokens['input_ids'],
                                                               attention_mask=question_tokens[
                                                                   'attention_mask']).last_hidden_state
            pooled_question_embeds = self.chat_util.mean_pool(question_embeds,
                                                              question_tokens['attention_mask']).cpu().numpy().squeeze(
                0)

        custom_answer_embeddings = []
        for precomputed_answer_embed in self.preprocessed_answers_embeddings:
            concatenated_query_answer_embeds = np.concatenate([pooled_question_embeds, precomputed_answer_embed,
                                                               np.abs(
                                                                   pooled_question_embeds - precomputed_answer_embed)],
                                                              axis=-1)
            custom_answer_embeddings.append(concatenated_query_answer_embeds)

        return np.array(custom_answer_embeddings)

    def find_top_n_unique_cosine_sim_bi_plus_cross_enc(self, query):
        self.chat_util.timestamp_log(
            "find_top_n_unique_cosine_sim_bi_plus_cross_enc - get_query_answers_embeddings_bi_encoder")

        query_answers_embeddings_bi_encoder = self.get_query_answers_embeddings_bi_encoder(
            self.bi_encoder_model.tokenizer,
            query)
        return self.find_top_n_unique_cosine_sim_bi_plus_cross_enc_aux(query_answers_embeddings_bi_encoder)

    def find_top_n_unique_cosine_sim_bi_plus_cross_enc_aux(self, custom_answer_embeddings):
        self.chat_util.timestamp_log("find_top_n_unique_cosine_sim - get cosine similarities")
        similarities = cosine_similarity(custom_answer_embeddings, self.preprocessed_questions_answers_embeddings)
        self.chat_util.timestamp_log("find_top_n_unique_cosine_sim - sorting")

        flat_similarities = similarities.flatten()

        seen_content = set()
        unique_top_indices = []
        unique_top_similarities = []

        sorted_indices = np.argsort(flat_similarities)[::-1]

        for idx in sorted_indices:
            matrix_idx = np.unravel_index(idx, similarities.shape)
            qa_pair = self.target_char_questions_and_answers[matrix_idx[1]]

            # сериализация для возможности сравнения
            qa_content = tuple(sorted(qa_pair.items()))

            if qa_content not in seen_content:
                seen_content.add(qa_content)
                unique_top_indices.append(matrix_idx[1])
                unique_top_similarities.append(flat_similarities[idx])

            if len(unique_top_indices) == self.constants.BI_ENCODER_TOP_N:
                break

        unique_top_question_answer_pairs = [self.target_char_questions_and_answers[idx] for idx in unique_top_indices]

        self.chat_util.timestamp_log("find_top_n_unique_cosine_sim - done")
        return unique_top_indices, unique_top_similarities, unique_top_question_answer_pairs

    def find_top_n_unique_l2_bi_plus_cross_enc(self, query):
        query_answers_embeddings_bi_encoder = self.get_query_answers_embeddings_bi_encoder(
            self.bi_encoder_model.tokenizer,
            query)
        return self.find_top_n_unique_l2_bi_plus_cross_enc_aux(query_answers_embeddings_bi_encoder)

    def find_top_n_unique_l2_bi_plus_cross_enc_aux(self, custom_query_answer_embeddings):
        self.chat_util.timestamp_log("find_top_n_unique_l2 - get indices")
        distances, indices = self.repository.preprocessed_questions_answers_embeddings_faiss_index.search(
            custom_query_answer_embeddings.astype('float32'),
            self.constants.BI_ENCODER_TOP_N * 2)
        self.chat_util.timestamp_log("find_top_n_unique_l2 - sorting")

        all_distances = distances.flatten()
        all_indices = indices.flatten()

        combined = list(zip(all_distances, all_indices))
        combined_sorted = sorted(combined, key=lambda x: x[0])

        unique_indices = set()
        top_n_unique = []

        for dist, idx in combined_sorted:
            if len(top_n_unique) == self.constants.BI_ENCODER_TOP_N:
                break

            if idx not in unique_indices:
                unique_indices.add(idx)
                top_n_unique.append((dist, idx))

        top_n_qa_pairs = [self.repository.target_char_questions_and_answers[idx] for _, idx in top_n_unique]
        self.chat_util.timestamp_log("find_top_n_unique_l2 - done")
        return top_n_qa_pairs, top_n_unique

    def find_top_n_unique_l2_psa_bi_plus_cross_enc(self, query):
        query_answers_embeddings_bi_encoder = self.get_query_answers_embeddings_bi_encoder(
            self.bi_encoder_model.tokenizer,
            query)
        return self.find_top_n_unique_l2_psa_bi_plus_cross_enc_aux(query_answers_embeddings_bi_encoder)

    def find_top_n_unique_l2_psa_bi_plus_cross_enc_aux(self, custom_query_answer_embeddings):
        custom_query_answer_embeddings_reduced = self.apply_pca_psa(custom_query_answer_embeddings,
                                                                    n_components=self.constants.PCA_COMPONENTS_COUNT)
        self.chat_util.timestamp_log("find_top_n_unique_l2_psa - get indices")
        distances, indices = self.repository.preprocessed_questions_answers_embeddings_faiss_psa_index.search(
            custom_query_answer_embeddings_reduced.astype('float32'),
            self.constants.BI_ENCODER_TOP_N * 2)
        self.chat_util.timestamp_log("find_top_n_unique_l2_psa - sorting")

        all_distances = distances.flatten()
        all_indices = indices.flatten()
        combined = list(zip(all_distances, all_indices))
        combined_sorted = sorted(combined, key=lambda x: x[0])

        unique_indices = set()
        top_n_unique = []
        self.chat_util.timestamp_log_psa("find unique")
        for dist, idx in combined_sorted:
            if len(top_n_unique) == self.constants.BI_ENCODER_TOP_N:
                break
            if idx not in unique_indices:
                unique_indices.add(idx)
                top_n_unique.append((dist, idx))
        top_n_qa_pairs = [self.repository.target_char_questions_and_answers[idx] for _, idx in top_n_unique]
        self.chat_util.timestamp_log("find_top_n_unique_l2_psa - done")
        return top_n_qa_pairs, top_n_unique

    def apply_pca_psa(self, embeddings, n_components=None):
        if n_components is None:
            n_components = self.constants.PCA_COMPONENTS_COUNT
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        return reduced_embeddings

    def find_similar_answers_cross_enc(self, query):
        self.cross_encoder_model = self.cross_encoder_model.to(self.constants.device)

        all_scores = []

        for i in range(0, len(self.target_char_answers), self.constants.CROSS_ENCODER_CHUNK_SIZE):
            chunk_answers = self.target_char_answers[i:i + self.constants.CROSS_ENCODER_CHUNK_SIZE]
            tokenized_pairs = self.cross_encoder_model.tokenizer([query] * len(chunk_answers), chunk_answers,
                                                                 padding=True, truncation=True, max_length=128,
                                                                 return_tensors="pt")
            tokenized_pairs = {k: v.to(self.constants.device) for k, v in tokenized_pairs.items()}

            with torch.no_grad():
                logits = self.cross_encoder_model(**tokenized_pairs).squeeze(-1)
                scores = torch.sigmoid(logits).cpu().numpy()

            all_scores.extend(scores)

            del tokenized_pairs, logits, scores
            torch.cuda.empty_cache()

        sorted_indices = np.argsort(all_scores)[::-1][:self.constants.CROSS_ENCODER_TOP_N]
        top_answers = [(self.target_char_answers[idx], all_scores[idx]) for idx in sorted_indices]

        return top_answers
