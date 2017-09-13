#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 22:57:00 2016

@author: vaishnavithangapalam
"""

import conf

f = open(conf.summarize_log,'w')



with open(conf.collect_log) as fp:
    for line in fp:
        f.write(line)
        
with open(conf.cluster_log) as fp1:
    for line1 in fp1:
        f.write(line1)        
        
with open(conf.classify_log) as fp2:
    for line1 in fp2:
        f.write(line1)  
        
f.close()