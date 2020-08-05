"""
A `Flask <http://flask.pocoo.org/>`_ server for serving predictions
from a single AllenNLP model. It also includes a very, very bare-bones
web front-end for exploring predictions (or you can provide your own).

For example, if you have your own predictor and model in the `my_stuff` package,
and you want to use the default HTML, you could run this like

```
python -m allennlp.service.server_simple \
    --archive-path allennlp/tests/fixtures/bidaf/serialization/model.tar.gz \
    --predictor machine-comprehension \
    --title "Demo of the Machine Comprehension Text Fixture" \
    --field-name question --field-name passage
```
"""
import json
import logging

from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from allennlp.common.checks import check_for_gpu
from allennlp.common.util import import_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from config import DEFAULT_PREDICTORS
from config import WEB_CONFIG
import bita.nlp.apis.nlp as nlp

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ServerError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        error_dict = dict(self.payload or ())
        error_dict['message'] = self.message
        return error_dict


def make_app() -> Flask:
    """
    Creates a Flask app that serves up the provided ``Predictor``
    along with a front-end for interacting with it.

    If you want to use the built-in bare-bones HTML, you must provide the
    field names for the inputs (which will be used both as labels
    and as the keys in the JSON that gets sent to the predictor).

    If you would rather create your own HTML, call it index.html
    and provide its directory as ``static_dir``. In that case you
    don't need to supply the field names -- that information should
    be implicit in your demo site. (Probably the easiest thing to do
    is just start with the bare-bones HTML and modify it.)

    In addition, if you want somehow transform the JSON prediction
    (e.g. by removing probabilities or logits)
    you can do that by passing in a ``sanitizer`` function.
    """


    app = Flask(__name__)  # pylint: disable=invalid-name

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error: ServerError) -> Response:  # pylint: disable=unused-variable
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.route('/')
    def index() -> Response:  # pylint: disable=unused-variable
        pass
        # if static_dir is not None:
        #     static_dir = os.path.abspath(static_dir)
        #     if not os.path.exists(static_dir):
        #         logger.error("app directory %s does not exist, aborting", static_dir)
        #         sys.exit(-1)
        # elif static_dir is None and field_names is None:
        #     print("Neither build_dir nor field_names passed. Demo won't render on this port.\n"
        #           "You must use nodejs + react app to interact with the server.")
        # if static_dir is not None:
        #     return send_file(os.path.join(static_dir, 'index.html'))
        # else:
        #     html = _html(title, field_names)
        #     return Response(response=html, status=200)

    @app.route('/segment', methods=['POST', 'OPTIONS', 'GET'])
    def segment() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.args
        prediction = nlp.segment(data["sentence"], ispos=True)
        return jsonify({"res": prediction})

    @app.route('/ner', methods=['POST', 'OPTIONS', 'GET'])
    def ner() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.args
        prediction = nlp.ner(data["sentence"])

        return jsonify({"res": prediction})

    @app.route('/sentiment', methods=['POST', 'OPTIONS', 'GET'])
    def sentiment() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.args
        prediction = nlp.sentiment(data["sentence"])
        return jsonify({"res": prediction})

    @app.route('/classify', methods=['POST', 'OPTIONS', 'GET'])
    def classify() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.args
        prediction = nlp.classify(data["sentence"])
        return jsonify({"res": prediction})

    @app.route('/similarity', methods=['POST', 'OPTIONS', 'GET'])
    def similarity() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.args
        prediction = nlp.similarity([data["premise"], data["hypothesis"]])
        return jsonify({"res": prediction})

    @app.route('/srl', methods=['POST', 'OPTIONS', 'GET'])
    def srl() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)
        data = request.args
        import bita.nlp.apis.nlp as nlp
        prediction = nlp.srl(data["sentence"])
        return jsonify({"res": prediction})

    @app.route('/uni_dp', methods=['POST', 'OPTIONS', 'GET'])
    def uni_dp() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.args
        prediction = nlp.uni_dp(data["sentence"])
        return jsonify({"res": prediction})

    @app.route('/sdp', methods=['POST', 'OPTIONS', 'GET'])
    def sdp() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.args
        prediction = nlp.sdp(data["sentence"])
        return jsonify({"res": prediction})

    @app.route('/<path:path>')
    def static_proxy(path: str) -> Response:  # pylint: disable=unused-variable
            raise ServerError("static_dir not specified", 404)
    return app


def _get_predictor(**params) -> Predictor:
    for package_name in params["include_package"]:
        import_submodules(package_name)
    cuda_device = params["cuda_device"] if "cuda_device" in params else -1
    weights_file = params["weights_file"] if "weights_file" in params else ""
    overrides = params["overrides"] if "overrides" in params else ""
    check_for_gpu(cuda_device)
    archive = load_archive(params["model_file"],
                           weights_file=weights_file,
                           cuda_device=cuda_device,
                           overrides=overrides)

    return Predictor.from_archive(archive, params["predictor"])


def main():
    # Executing this file with no extra options runs the simple service with the bidaf test fixture
    # and the machine-comprehension predictor. There's no good reason you'd want
    # to do this, except possibly to test changes to the stock HTML).

    app = make_app()
    CORS(app)

    http_server = WSGIServer(('0.0.0.0', WEB_CONFIG.PORT), app)
    print(f"Model loaded, serving demo on port {WEB_CONFIG.PORT}")
    http_server.serve_forever()

#
# HTML and Templates for the default bare-bones app are below
#


if __name__ == "__main__":
    main()
