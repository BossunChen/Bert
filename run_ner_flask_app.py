# -*- coding: utf-8 -*-
"""
__author__ = liuxiangyu
__mtime__ = 2020/11/26 14:33
"""
from flask import Flask


def run_as_blueprint():
    from new_ner_flask import NerApp
    ner_app = NerApp()


    app = Flask(__name__)

    app.register_blueprint(ner_app.get_flask_blueprint())



    app.run(host='0.0.0.0', port=8989, debug=True)

# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--port', action='store', help='App port number to run , default is 5000', type=str,
#                         default=15000)
#     parser.add_argument('--ip', action='store', help='Manuel assign ip address ,default is 0.0.0.0', type=str,
#                         default="0.0.0.0")
#     parser.add_argument('--debug', action='store', help='debug', default=False)
#     return parser.parse_args()


if __name__ == '__main__':
    run_as_blueprint()