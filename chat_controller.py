import threading
from flask import Flask, jsonify, request
from pyngrok import ngrok


class ChatController:
    def __init__(self, chat_service, constants):
        self.chat_service = chat_service
        self.app = Flask(__name__)
        self.configure_routes()
        self.constants = constants

    def configure_routes(self):
        @self.app.route("/top_n_unique_cosine_sim_bi_plus_cross_enc", methods=["POST"])
        def find_top_n_unique_cosine_sim_bi_plus_cross_enc():
            query = request.json.get("query", "")
            user = request.json.get("user", "default_user")
            response = self.chat_service.find_top_n_unique_cosine_sim_bi_plus_cross_enc(query, user)
            return jsonify(response=response)

        @self.app.route("/top_n_unique_l2_bi_plus_cross_enc", methods=["POST"])
        def find_top_n_unique_l2_bi_plus_cross_enc():
            query = request.json.get("query", "")
            user = request.json.get("user", "default_user")
            response = self.chat_service.find_top_n_unique_l2_bi_plus_cross_enc(query, user)
            return jsonify(response=response)

        @self.app.route("/top_n_unique_l2_psa_bi_plus_cross_enc", methods=["POST"])
        def find_top_n_unique_l2_psa_bi_plus_cross_enc():
            query = request.json.get("query", "")
            user = request.json.get("user", "default_user")
            response = self.chat_service.find_top_n_unique_l2_psa_bi_plus_cross_enc(query, user)
            return jsonify(response=response)

        @self.app.route("/top_n_unique_answers_cross_enc", methods=["POST"])
        def find_top_n_unique_answers_cross_enc():
            query = request.json.get("query", "")
            user = request.json.get("user", "default_user")
            response = self.chat_service.find_top_n_unique_answers_cross_enc(query, user)
            return jsonify(response=response)

    def run(self):
        public_url = ngrok.connect(5000).public_url
        print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 5000))
        self.app.config["BASE_URL"] = public_url

        # запустить в отдельном потоке
        threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()
