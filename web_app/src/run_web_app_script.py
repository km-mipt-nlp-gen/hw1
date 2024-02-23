import sys
import logging
from joblib import load
import torch

from constants_module import Constants
from chat_util_module import ChatUtil
from chat_service_module import ChatService
from chat_repository_module import ChatRepository
from chat_service_accelerator_module import ChatServiceAccelerator
from chat_controller_module import ChatController


def run_web_app():
    constants = Constants()
    chat_util = ChatUtil(logging.DEBUG, constants)

    bi_encoder_model, cross_encoder_model = initialize_models(constants, chat_util)

    target_char_questions_and_answers = load(constants.TARGET_CHAR_PROCESSED_QA_PATH)
    target_char_answers = load(constants.TARGET_CHAR_PROCESSED_ANSWERS_PATH)

    chat_msg_history = []

    chat_repository = None

    if constants.IS_EMBEDDINGS_USED:
        chat_repository = ChatRepository(
            chat_msg_history, target_char_questions_and_answers, target_char_answers,
            bi_encoder_model, cross_encoder_model, None, chat_util,
            constants.TARGET_CHAR_QA_PAIRS_EMBEDDINGS_PATH,
            constants.TARGET_CHAR_ANSWERS_EMBEDDINGS_PATH,
            constants.TARGET_CHAR_QA_PAIRS_FAISS_INDEX_PATH,
            constants.TARGET_CHAR_QA_PAIRS_FAISS_PSA_INDEX_PATH
        )
    else:
        chat_service_accelerator = ChatServiceAccelerator(
            bi_encoder_model, cross_encoder_model, target_char_questions_and_answers, target_char_answers,
            constants, chat_util)
        chat_repository = ChatRepository(
            chat_msg_history, target_char_questions_and_answers, target_char_answers,
            bi_encoder_model, cross_encoder_model, chat_service_accelerator, chat_util)

    chat_service = ChatService(chat_msg_history, chat_repository, constants, chat_util)

    chat_controller = ChatController(chat_service, constants, chat_util)
    chat_controller.init_conf()

    chat_controller.run()


def initialize_models(constants, chat_util):
    sys.path.append(constants.WEB_APP_SRC_PATH)
    from models_zoo_module import SiameseBiEncoder, CrossEncoder

    bi_encoder_model = SiameseBiEncoder(constants, chat_util).to(constants.DEVICE)
    bi_encoder_path = constants.BI_ENCODER_MODEL_PATH
    bi_encoder_model.load_state_dict(torch.load(bi_encoder_path, map_location=constants.DEVICE))

    cross_encoder_model = CrossEncoder(constants).to(constants.DEVICE)
    cross_encoder_path = constants.CROSS_ENCODER_MODEL_PATH
    cross_encoder_model.load_state_dict(torch.load(cross_encoder_path, map_location=constants.DEVICE))

    return bi_encoder_model, cross_encoder_model

# Разкомментировать для запуска вне colab
# if __name__ == "__main__":
#     initialize_and_run_application()
