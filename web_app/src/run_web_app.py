import logging
from joblib import load
import torch
import SiameseBiEncoder, CrossEncoder  # TODO
from chat_constants import Constants
from chat_util import ChatUtil
from chat_service import ChatService
from chat_repository import ChatRepository
from chat_service_accelerator import ChatServiceAccelerator
from chat_controller import ChatController


def run_web_app():
    constants = Constants()

    bi_encoder_model = SiameseBiEncoder().to(constants.device)
    bi_encoder_path = '/content/drive/MyDrive/docs/keepForever/mipt/nlp/hw1_4sem/tmp/models/SiameseBiEncoder_20240218_005050.pth'  # TODO
    bi_encoder_model.load_state_dict(torch.load(bi_encoder_path, map_location=constants.device))

    cross_encoder_model = CrossEncoder().to(constants.device)
    cross_encoder_path = '/content/drive/MyDrive/docs/keepForever/mipt/nlp/hw1_4sem/tmp/models/CrossEncoder_20240221_165603.pth'  # TODO
    cross_encoder_model.load_state_dict(torch.load(cross_encoder_path, map_location=constants.device))

    target_char_questions_and_answers = load(constants.TARGET_CHAR_PROCESSED_QA_PATH)
    target_char_answers = load(constants.TARGET_CHAR_PROCESSED_ANSWERS_PATH)

    chat_util = ChatUtil(logging.DEBUG, constants)

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

    chat_controller = ChatController(chat_service, constants)
    chat_controller.init_conf()

    chat_controller.run()

# Разкомментировать для запуска вне colab
# if __name__ == "__main__":
#     initialize_and_run_application()
