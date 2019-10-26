#!/usr/bin/python
# -*- coding: utf-8 -*-

import healthtrends

DIR = './'

queries = ['tofu', 'exercise']

gt = healthtrends.TrendsSession(api_key='xxx')
gt.request_trends(term_list=queries, geo_level='country', geo_id='US')
gt.save_to_csv(directory=DIR, fname='healthy_trends.csv')
