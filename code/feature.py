#! /usr/bin/env python
"""Mysql Feature class"""

import peewee as pw
import cPickle as pickle


connect_kwargs = {'host':'127.0.0.1', 'user':'nialzcom_ml', 'passwd':'o4v:4GS}', 'port':9870}
mysql_db = pw.MySQLDatabase('nialzcom_ML', **connect_kwargs)

class PickleField(pw.BlobField):
    """Pickls and unpickles a blob field"""
    def db_value(self, value):
        """convert from python (numpy array) to database (pickled)"""
        if value != None:
            return pickle.dumps(value, protocol=2)
        else:
            return value

    def python_value(self, value):
        """convert from data base (pickled) to python (numpy array)"""
        if value != None:
            return pickle.loads(value)
        else:
            return value

class Feature(pw.Model):
    """Mysql Feature class"""
    name = pw.CharField(unique=True)
    label = pw.IntegerField(null=True)

    gray_hist = PickleField()
    red_hist = PickleField()
    green_hist = PickleField()
    blue_hist = PickleField()
    hue_hist = PickleField()
    saturation_hist = PickleField()
    value_hist = PickleField()
    pca = PickleField(null=True)

    class Meta:
        """Contains the database"""
        database = mysql_db
