#! /usr/bin/env python
"""Mysql Feature class"""

import peewee as pw
import cPickle as pickle


connect_kwargs = {'host':'', 'user':'', 'passwd':'', 'port':}
mysql_db = pw.MySQLDatabase('nialzcom_ML', **connect_kwargs)

class PickleField(pw.BlobField):
    """Pickls and unpickles a blob field"""
    def db_value(self, value):
        """convert from python (numpy array) to database (pickled)"""
        return pickle.dumps(value, protocol=2)

    def python_value(self, value):
        """convert from data base (pickled) to python (numpy array)"""
        return pickle.loads(value)

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

    class Meta:
        """Contains the database"""
        database = mysql_db
