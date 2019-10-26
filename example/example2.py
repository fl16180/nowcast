#!/usr/bin/python
# -*- coding: utf-8 -*-

import healthtrends

# lazy fix for utf-8 encoding for accents and complex characters in term list
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# Here we assume API key is defined as variable 'APIKEY' in separate config.py file
# for security when making code publicly available
import config


# start session
gt = healthtrends.TrendsSession(api_key=config.APIKEY)

# download from Rio de Janeiro state
queries = ['vacinação gripe']
gt.request_trends(term_list=queries, geo_level='region', geo_id='BR-RJ')
gt.save_to_csv(directory='/home/data/googletrends/BR', fname='vaccine.csv')

# download from Greater Boston metropolitan area
queries = ['how long contagious', 'flu symptoms']
gt.request_trends(term_list=queries, geo_level='dma', geo_id='503')
gt.save_to_csv(full_path='/home/data/googletrends/US/vaccine.csv')
