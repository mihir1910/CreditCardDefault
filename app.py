from flask import Flask, render_template,request
import  sklearn
import joblib
app=Flask(__name__)
model=joblib.load("credit.pkl")
@app.route('/')
def home ():
    a=request.form.get('email adress')
    print(a)
    return render_template('home.html', **locals())

@app.route('/submit',methods=['post'])
def submit():
    a=eval(request.form.get('limit_bal'))
    b=eval(request.form.get('sex'))
    c=eval(request.form.get('education'))
    d=eval(request.form.get('marriage'))
    e=eval(request.form.get('age'))
    f=eval(request.form.get('pay_0'))
    g=eval(request.form.get('pay_2'))
    h=eval(request.form.get('pay_4'))
    i=eval(request.form.get('pay_5'))
    j=eval(request.form.get('pay_6'))
    k=eval(request.form.get('prev_payment'))
    res1=model.predict([[a,b,c,d,e,f,g,h,i,j,k]])
    if res1[0]==1:
        return "Defaulter"
    else:
        return "Not a Defaulter"
    # result=model.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m,n]])[0]
    # return render_template('home.html',**locals())


app.run(debug=True)
