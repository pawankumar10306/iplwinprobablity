from flask import Flask,request,render_template,jsonify
import pickle
import pandas as pd

final=pd.read_csv('final.csv')
app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def home():
  batting_team=sorted(final['batting_team'].unique())
  city=sorted(final['city'].unique())
  return render_template('index.html',city=city,batting_team=batting_team)


@app.route('/predict',methods=['POST'])
def predict():
    model=pickle.load(open('pipe.pkl','rb'))
    batting_team=request.form.get('batting_team')
    bowling_team=request.form.get('bowling_team')
    city=request.form.get('city')
    score=request.form.get('score')
    overs=request.form.get('overs')
    target=request.form.get('target')
    wickets=request.form.get('wickets')
    
    runs_left = int(target) - int(score)
    balls_left = 120 - (int(overs)*6)
    wickets = 10 - int(wickets)
    crr = int(score)/int(overs)
    rrr = (runs_left*6)/balls_left

    input_query = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result = model.predict_proba(input_query)
    
    loss = result[0][0]
    win = result[0][1]
    return jsonify({'win':str(round(win*100)),'loss':str(round(loss*100))})
  
if __name__ == '__main__':
  app.run(debug=True)



