# -*- coding: utf-8 -*-
# __author : Bossun_Chen
# __time : 2021/12/9 下午 06:17

import json
from flask import Flask, request, Response, Blueprint
from flask_cors import CORS

from flask_predict import PredictModel

app = Flask(__name__)

class NerApp():

    app_blueprint = Blueprint('NerApp', __name__)

    def __init__(self):
        NerApp.predict_model = PredictModel()

    def get_flask_app(self, **kwargs):
        app = Flask("PneumoniaRecApp")
        app.config['SWAGGER'] = NerApp.SWAGGER_INFO
        app.register_blueprint(self.app_blueprint)
        CORS(app, supports_credentials=True)
        return app

    def get_flask_blueprint(self, **kwargs):
        return self.app_blueprint

    @app_blueprint.route('/nerhome')
    def home_page():
        return 'Welcome to use ner'

    @app_blueprint.route('/NER', methods=['post'])
    def ner_demo():

        # postman Body json用法
        json_data = request.get_json()

        # 更新
        keys_names = []
        for record in json_data:
            # 更新
            keys_name = []
            for keys in record:
                keys_name.append(keys)
            keys_names.append(keys_name)

        # 一行解决
        # keys_names= [[keys for keys in record]for record in json_data]

        ners = []
        for i, record in enumerate(json_data):

            ner = dict()

            # 更新
            if "eventId" not in keys_names[i]:
                get_eventId = ""
            else:
                get_eventId = record["eventId"]

            if "caseId" not in keys_names[i]:
                get_caseId = ""
            else:
                get_caseId = record["caseId"]

            if "travelContent" not in keys_names[i]:
                get_travelContent = ""
            else:
                get_travelContent = record["travelContent"]

            if "sourceId" not in keys_names[i]:
                get_sourceId = ""
            else:
                get_sourceId = record["sourceId"]

            # get_eventId = record["eventId"]
            # get_caseId = record["caseId"]
            # get_sourceId = record["sourceId"]
            # get_travelContent = record["travelContent"]

            # 初始化类
            if get_travelContent == "":
                n = 0
            else:
                predict_ = NerApp.predict_model.predict(get_travelContent)


                dates = predict_[1]['date']
                names = predict_[0]['name']
                locations = predict_[1]['location']
                licenses = predict_[0]['license']

                n = max(len(dates), len(names), len(locations), len(licenses))

                # 填充到一样的长度
                dates.extend([""] * (n - len(dates)))
                names.extend([""] * (n - len(names)))
                locations.extend([""] * (n - len(locations)))
                licenses.extend([""] * (n - len(licenses)))


            if n==0:
                n=1
                for i in range(n):
                    ner = {
                        'eventId': get_eventId,
                        'caseId': get_caseId,
                        'sourceId': get_sourceId,
                        'travelTime': "",  # travelTime
                        'digPersonName': "",  # digPersonName # 界定标准
                        'digPlaceName': "",  # digPlaceName
                        'digTrafficTool': ""  # digTrafficTool
                    }
                    ners.append(ner)
            else:
                for i in range(n):
                    ner = {
                        'eventId': get_eventId,
                        'caseId': get_caseId,
                        'sourceId': get_sourceId,
                        'travelTime': dates[i],  # travelTime
                        'digPersonName': names[i],  # digPersonName # 界定标准
                        'digPlaceName': locations[i],  # digPlaceName
                        'digTrafficTool': licenses[i]  # digTrafficTool
                    }
                    ners.append(ner)

        return Response(json.dumps({'code': 1, 'msg': "执行成功", 'data': ners}), mimetype='application/json')
