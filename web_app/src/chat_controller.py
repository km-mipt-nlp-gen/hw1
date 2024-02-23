import threading
from flask import Flask, jsonify, request
from pyngrok import ngrok, conf
import getpass


class ChatController:
    def __init__(self, chat_service, constants):
        self.chat_service = chat_service
        self.app = Flask(__name__)
        self.configure_routes()
        self.constants = constants

    def configure_routes(self):
        @self.app.route("/top_cos_sim_bi_cr", methods=["POST"])
        def find_top_n_unique_cosine_sim_bi_plus_cross_enc():
            try:
                query = request.json.get("query", "")
                user = request.json.get("user", "default_user")
                response = self.chat_service.find_top_n_unique_cosine_sim_bi_plus_cross_enc(query, user)
                return jsonify(response=response), 200
            except Exception as e:
                return jsonify(self.get_error(
                    "Ошибка получения лучшего ответа (метрика cosine sim, архитектура bi-encoder cross-encoder)",
                    e)), 500

        @self.app.route("/top_l2_bi_cr", methods=["POST"])
        def find_top_n_unique_l2_bi_plus_cross_enc():
            try:
                query = request.json.get("query", "")
                user = request.json.get("user", "default_user")
                response = self.chat_service.find_top_n_unique_l2_bi_plus_cross_enc(query, user)
                return jsonify(response=response), 200
            except Exception as e:
                return jsonify(
                    self.get_error("Ошибка получения лучшего ответа (метрика l2, архитектура bi-encoder cross-encoder)",
                                   e)), 500

        @self.app.route("/top_l2_psa_bi_cr", methods=["POST"])
        def find_top_n_unique_l2_psa_bi_plus_cross_enc():
            try:
                query = request.json.get("query", "")
                user = request.json.get("user", "default_user")
                response = self.chat_service.find_top_n_unique_l2_psa_bi_plus_cross_enc(query, user)
                return jsonify(response=response), 200
            except Exception as e:
                return jsonify(self.get_error(
                    "Ошибка получения лучшего ответа (метрика l2 psa, архитектура bi-encoder cross-encoder)", e)), 500

        @self.app.route("/top_cr", methods=["POST"])
        def find_top_n_unique_answers_cross_enc():
            try:
                query = request.json.get("query", "")
                user = request.json.get("user", "default_user")
                response = self.chat_service.find_top_n_unique_answers_cross_enc(query, user)
                return jsonify(response=response), 200
            except Exception as e:
                return jsonify(self.get_error("Ошибка получения лучшего ответа (архитектура cross-encoder)", e)), 500

        @self.app.route("/clear", methods=["DELETE"])
        def clear_chat_msg_history():
            try:
                self.chat_service.clear_chat_msg_history()
                return jsonify({"сообщение": "Чат очищен"}), 200
            except Exception as e:
                return jsonify(self.get_error("Ошибка очистки чата", e)), 500

        @self.app.route("/chat", methods=["GET"])
        def get_chat_msg_history():
            try:
                chat_msg_history = self.chat_service.chat_msg_history()
                return jsonify(response=chat_msg_history), 200
            except Exception as e:
                return jsonify(self.get_error("Ошибка загрузки чата", e)), 500

    @staticmethod
    def get_error(message, exception):
        return {
            "сообщение": message,
            "детали": str(exception)
        }

    def init_conf(self):
        conf.get_default().auth_token = getpass.getpass()

    def run(self):
        tunnels = ngrok.get_tunnels()
        for tunnel in tunnels:
            ngrok.disconnect(tunnel.public_url)

        ngrok.kill()
        public_url = ngrok.connect(5000).public_url
        print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 5000))
        self.app.config["BASE_URL"] = public_url

        # запустить в отдельном потоке
        threading.Thread(target=self.app.run, kwargs={"use_reloader": False}).start()