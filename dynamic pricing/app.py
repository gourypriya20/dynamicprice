from flask import Flask, render_template, redirect, url_for,  request, session

import csv
import pickle
from model.cloth import read
from model.car import car_prediction
from model.oven import appliance_oven
from model.shoes import shoe
from model.bag import bag
app = Flask(__name__)
app.secret_key = 'your_secret_key'
        
users = {'user1': 'password123'}

@app.route('/',methods=["GET", "POST"])
def sign_in():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return "Invalid credentials. Please try again."
    return render_template('sign.html')

@app.route('/index',methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route('/shop', methods=["GET", "POST"])
def shop():
    return render_template("shop.html")

@app.route('/about',methods=["GET", "POST"])
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact',methods=["GET", "POST"])
@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/cart')
def cart():
    return render_template('cart.html')
    
    
@app.route('/cloth',methods=["GET", "POST"])
def product_display():
    if request.method == "POST":
        predict_result =  read()
        return render_template('cloth.html', result = predict_result)
    else:
        return render_template('cloth.html')
    
    
@app.route('/cars',methods=["GET", "POST"])
def car_display():
    if request.method == "POST":
        predict_result =  car_prediction()
        return render_template('cars.html', result = predict_result)
    else:
        return render_template('cars.html')


@app.route('/oven',methods=["GET", "POST"])
def oven_display():
    if request.method == "POST":
        predict_result =  appliance_oven()
        return render_template('oven.html', result = predict_result)
    else:
        return render_template('oven.html')

@app.route('/shoe',methods=["GET", "POST"])
def shoes_display():
    if request.method == "POST":
        predict_result =  shoe()
        return render_template('shoes.html', result = predict_result)
    else:
        return render_template('shoes.html')

@app.route('/bag',methods=["GET", "POST"])
def bag_display():
    if request.method == "POST":
        predict_result =  bag()
        return render_template('bag.html', result = predict_result)
    else:
        return render_template('bag.html')