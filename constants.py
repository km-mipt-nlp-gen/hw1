class Constants:

    def __init__(self):
        pass

    # Пути
    WORKSPACE_PATH = '/content/drive/MyDrive/docs/keepForever/mipt/nlp/hw1_4sem/'
    WORKSPACE_TMP = WORKSPACE_PATH + '/tmp/'
    INPUT_DATA_DIR_PATH = WORKSPACE_PATH + 'input_data/'
    PROCESSED_DATA_DIR_PATH = WORKSPACE_PATH + 'processed_data/'
    THE_SIMPS_CSV_PATH = INPUT_DATA_DIR_PATH + 'script_lines.csv'

    TARGET_CHAR_PROCESSED_QA_PATH = PROCESSED_DATA_DIR_PATH + 'target_char_qa_pairs.joblib'

    # Целевой персонаж
    LISA_ID = 9
    LISA_FULL_NAME = 'Lisa Simpson'
    LISA_LC_NAME = 'lisa'

    # Столбцы
    EPISODE_ID_COL = 'episode_id'
    PREV_EPISODE_ID_COL = 'prev_episode_id'
    NUMBER_COL = 'number'

    SPEAKING_LINE_COL = 'speaking_line'
    NORM_TEXT_COL = 'normalized_text'
    RAW_CHAR_TEXT_COL = 'raw_character_text'

    CHAR_ID_COL = 'character_id'
    PREMISE_CHAR_ID_COL = 'premise_char_id'

    LOC_ID_COL = 'location_id'
    PREV_LOC_ID_COL = 'prev_location_id'

    SPOKEN_WORDS_COL = 'spoken_words'
    PREMISE_COL = 'premise'
    TARGET_CHAR_ANSWER_COL = 'target_char_answer_col'

    SAME_LOC_ID_COL = 'same_location_id_dialog'
    SAME_EPISODE_ID_COL = 'same_episode_id'

    RAW_TEXT_COL = 'raw_text'

    LABEL_COL = 'label'
    INVALID_QA_MARK = 0
    VALID_QA_MARK = 1

    # Столбцы для сортировки
    SIMPS_DF_SORT_BY_COLS = [EPISODE_ID_COL, NUMBER_COL]

    ''' Прочие константы '''

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # тренировка
    SEED = 14
    EPOCHS_OVERFIT_COUNT = 1
    BATCH_SIZE = 16

    # вывод
    BI_ENCODER_TOP_N = 8
    GPU_FAISS_INDEX = True
    CROSS_ENCODER_TOP_N = 8

    PROC_COUNT = cpu_count()
    print(f"Число процессов для использования: {PROC_COUNT}")

    # воспроизводимость
    seed = SEED

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True  # для воспроизводимости
    torch.backends.cudnn.benchmark = False
