from flask import Flask,render_template,request
import flask
import time
app=Flask(__name__)

@app.route('/')
def dyn_page():
    return render_template('home.html')
@app.route('/medbot',methods=['GET','POST'])
def hello():
    return "Helllo"
@app.route('/medbot2', methods=['GET','POST'])
def sub3():
    import bot.py as bot
    return bot

if __name__ == '__main__':
    app.run(debug=True)