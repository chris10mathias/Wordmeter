from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
#from Code.Eivatable1 import *
from Code.Eiva import *
import pandas as pd

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
cors = CORS(app)


@app.route('/table1', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def Eivatable1():
    req_dict = request.get_json(force=True, silent=True)
    response=aspect_table(req_dict)
    return jsonify(response)

@app.route('/table2', methods=['GET'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def Eivatable2():
    response=order_table()
    return jsonify(response) 

@app.route('/barchart1', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def IntentSentiment():
    req_dict = request.get_json(force=True, silent=True)
    response=bar_chart_sentiment(req_dict)
    return jsonify(response) 

@app.route('/barchart2', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def Sentimentcp():
    req_dict = request.get_json(force=True, silent=True)
    response=sentimentcp(req_dict)
    return jsonify(response) 

@app.route('/barchart3', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def Sentimentproduct():
    req_dict = request.get_json(force=True, silent=True)
    response=sentimentproduct(req_dict)
    return jsonify(response)

@app.route('/topbar', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def Topbar_values():
    req_dict = request.get_json(force=True, silent=True)
    response=topbar_values(req_dict)
    return jsonify(response)  

@app.route('/doughnutchart1', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def Doughnutchart_user():
    req_dict = request.get_json(force=True, silent=True)
    response=doughnutchart_user(req_dict)
    return jsonify(response)   

@app.route('/agentbarchart1', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def agentcolumnchart1():
    req_dict = request.get_json(force=True, silent=True)
    response=Agentbarchart(req_dict)
    return jsonify(response)     
    
@app.route('/agenttree1', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def agenttree1():
    req_dict = request.get_json(force=True, silent=True)
    response=Agenttree(req_dict)
    return jsonify(response)

@app.route('/agenttable1', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def agenttable1():
    req_dict = request.get_json(force=True, silent=True)
    response= neg_scoretable(req_dict)
    return jsonify(response)

@app.route('/orderstable1', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def orderstable1():
    req_dict = request.get_json(force=True, silent=True)
    response= new_orderstable(req_dict)
    return jsonify(response)  

@app.route('/customertable1', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def customertable1():
    req_dict = request.get_json(force=True, silent=True)
    response= customer_table(req_dict)
    return jsonify(response)

@app.route('/trendsbarchart', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def Trends_barcharts():
    req_dict = request.get_json(force=True, silent=True)
    response= trends_barchart(req_dict)
    return jsonify(response)   

@app.route('/trendslinechart', methods=['POST'])
@cross_origin(headers=['Content-Type', 'Authorization'])
def Trends_linecharts():
    req_dict = request.get_json(force=True, silent=True)
    response= trends_linechart(req_dict)
    return jsonify(response)    
    
if __name__ == "__main__":
    app.run()     