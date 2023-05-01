from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

#Load pickel file
model = pickle.load(open('rf_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    Item_Identifier = request.form['Item ID']
    Outlet_Identifier = request.form['Outlet ID']
    #=====================================================

    features=[]
    # ============================================================================
    Item_Weight = float(request.form['Weight'])
    Item_Visibility = float(request.form['Range 0.6-1.8'])
    Item_MRP = float(request.form['Item MRP'])
    Outlet_Year = int(2013 - int(request.form['Year']))

    features.append(Item_Weight)
    features.append(Item_Visibility)
    features.append(Item_MRP)
    features.append(Outlet_Year)
    #============================================================================

    item_fat_content=request.form['Item Fat Content']

    if (item_fat_content== 'Low Fat'):
        item_fat_content = 1,0,0
    elif (item_fat_content== 'Non Edible'):
        item_fat_content = 0,1,0
    else:
        item_fat_content = 0,0,1

    Item_Fat_Content_1,Item_Fat_Content_2,Item_Fat_Content_3 = item_fat_content

    features.append(Item_Fat_Content_1)
    features.append(Item_Fat_Content_2)
    features.append(Item_Fat_Content_3)

    #============================================================================



    Item_Type_Combined = request.form['Item Type']

    if (Item_Type_Combined == "Drinks"):
        Item_Type_Combined = 1, 0, 0
    elif (Item_Type_Combined == "Foods"):
        Item_Type_Combined = 0, 1, 0
    else:
        Item_Type_Combined = 0, 0, 1

    Item_Type_Combined_1, Item_Type_Combined_2, Item_Type_Combined_3 = Item_Type_Combined

    features.append(Item_Type_Combined_1)
    features.append(Item_Type_Combined_2)
    features.append(Item_Type_Combined_3)

    # ============================================================================
    Outlet_Size  = request.form['Size']

    if (Outlet_Size == 'High'):
        Outlet_Size = 1,0,0,0
    elif (Outlet_Size == 'Medium'):
        Outlet_Size = 0,1,0,0
    elif (Outlet_Size == 'Small'):
        Outlet_Size = 0,0,1,0
    else:
        Outlet_Size = 0,0,0,1

    Outlet_Size_1, Outlet_Size_2,Outlet_Size_3, Outlet_Size_4 = Outlet_Size

    features.append(Outlet_Size_1)
    features.append(Outlet_Size_2)
    features.append(Outlet_Size_3)
    features.append(Outlet_Size_4)

    # ============================================================================
    Outlet_Location_Type = request.form['Location Type']

    if (Outlet_Location_Type == 'Tier 1'):
        Outlet_Location_Type = 1,0,0
    elif (Outlet_Location_Type == 'Tier 2'):
        Outlet_Location_Type = 0,1,0
    else:
        Outlet_Location_Type = 0,0,0

    Outlet_Location_Type_1,Outlet_Location_Type_2 ,Outlet_Location_Type_3= Outlet_Location_Type

    features.append(Outlet_Location_Type_1)
    features.append(Outlet_Location_Type_2)
    features.append(Outlet_Location_Type_3)

    # ============================================================================
    Outlet_Type = request.form['Outlet Type']

    if (Outlet_Type == 'Grocery Store'):
        Outlet_Type = 1,0,0,0
    elif (Outlet_Type == 'Supermarket Type1'):
        Outlet_Type = 0,1,0,0
    elif (Outlet_Type == 'Supermarket Type2'):
        Outlet_Type = 0,0,1,0
    elif (Outlet_Type == 'Supermarket Type3'):
        Outlet_Type = 0,0,0,1

    Outlet_Type_1, Outlet_Type_2, Outlet_Type_3,Outlet_Type_4 = Outlet_Type

    features.append(Outlet_Type_1)
    features.append(Outlet_Type_2)
    features.append(Outlet_Type_3)
    features.append(Outlet_Type_4)

    # ============================================================================
    features_value = [np.array(features)]

    features_name = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Years_of_Operation',
       'Low Fat', 'Non Edible', 'Regular', 'Drinks', 'Foods',
       'Non Consumables', 'High', 'Medium', 'Small', 'Unknown', 'Tier 1',
       'Tier 2', 'Tier 3', 'Grocery Store', 'Supermarket Type1',
       'Supermarket Type2', 'Supermarket Type3']

    print("Data Frame", features)

    df = pd.DataFrame(features_value, columns=features_name)

    myprd = model.predict(df)
    output=round(myprd[0],2)

    return render_template('result.html',prediction = output,Item_Identifier = Item_Identifier, Outlet_Identifier =Outlet_Identifier)

if __name__ == '__main__':
    app.run(debug=True)