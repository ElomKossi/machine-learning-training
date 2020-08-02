#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:58:46 2020

@author: joke
"""

import pandas as pd
import numpy as np
#import cx_Oracle as cx
import mysql.connector

dataset_csv = pd.read_csv('credit_immo.csv')

dataset_json = pd.read_json('credit_immo.json')

dataset_excel = pd.read_excel('credit_immo.xls')

"""
#connexion à la base de donnée oracle
ip = "192.168.0.12"
port = 1521
#service = "ORCLCDB.localdomain"

dns_tns = cx.makedsn(ip, port, service_name = "ORCLCDB.localdomain")

connexion = cx.connect("IMMO", "pwd", dns_tns)

query = "SELECT * FROM credit;"

dataset_sgbdr = pd.read_sql(query, con=connexion)
"""

#mysql
# Connect to server
cnx = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="sozaca",
    database='spiderTest')


# Get a cursor
#cursor = cnx.cursor()

# Execute a query
query = "SELECT * FROM CREDIT;"
#cursor.execute(query)

dataset_sql = pd.read_sql(query, con=cnx)


