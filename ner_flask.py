# -*- coding: utf-8 -*-
# __author : Bossun_Chen
# __time : 2021/12/3 下午 05:21

import json
from flask import Flask, request, Response
from run import predict

app = Flask(__name__)


@app.route('/NER',methods=['post'])
def ner_demo():

    # postman Body json用法
    json_data = request.get_json()
    ners = []
    for record in json_data:
        get_eventId = record["eventId"]
        get_caseId = record["caseId"]
        get_sourceId = record["sourceId"]
        get_travelContent = record["travelContent"]

        # 初始化类
        predict_ = predict(get_travelContent)

        ner = {
                'eventId': get_eventId,
                'caseId': get_caseId,
                'sourceId': get_sourceId,
                'travelTime': predict_[1]['date'], #travelTime
                'digPersonName': predict_[0]['name'], #digPersonName
                'digPlaceName': predict_[1]['location'], #digPlaceName
                'digTrafficTool': predict_[0]['license'] #digTrafficTool
            }
        ners.append(ner)

    return Response(json.dumps({'code': 1, 'msg': "执行成功", 'data': ners}), mimetype='application/json')

if __name__ == '__main__':
       app.debug = True

       app.run(host='0.0.0.0', port=8989)