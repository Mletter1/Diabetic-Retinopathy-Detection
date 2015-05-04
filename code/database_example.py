from feature import Feature, mysql_db
import numpy as np

#connect to Nialls's through ssh tunnel

#At start, connect (MAKE SURE YOU HAVE UPDATED feature.py WITH DB
# SETTINGS)
mysql_db.connect()

#Get training
query = Feature.select().where(Feature.label.is_null(False))
gray_hists = [] # See feature.py for fields in table
red_hists = []
labels = []
for feature in query:
    gray_hists.append(feature.gray_hist)
    red_hists.append(feature.red_hist)
    labels.append(feature.label)

gray_train = np.vstack(gray_hists)
red_train = np.vstack(red_hists)
gray_and_red_train = np.hstack([gray_train, red_train])

#Fit using features
# model.fit(gray_train, np.array(labels))

#Get test
query = Feature.select().where(Feature.label.is_null(True)).order_by(Feature.name)
gray_hists = [] # See feature.py fields in table
red_hists = []
names = []
for feature in query:
    names.append(feature.name)
    gray_hists.append(feature.gray_hist)
    red_hists.append(feature.red_hist)

gray_test = np.vstack(gray_hists)
red_test = np.vstack(red_hists)
gray_and_red_test = np.hstack([gray_train, red_train])

# Predict test labels
# predicted = model.predict(gray_test)

# Output
# output(names, predicted)
    
#At end, close
mysql_db.close()
